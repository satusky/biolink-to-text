"""Thin wrapper over the Biolink Model Toolkit (``bmt``).

Keeps schema access (predicate/qualifier validation, enum permissible values,
candidate predicates, slot descriptions, label humanization) in one place and
behind a lazily-initialized singleton so the model is loaded at most once per
process.
"""

from __future__ import annotations

from functools import lru_cache

from bmt import Toolkit

# Enum-valued qualifier slots offered for extraction. The backing enum is read
# from each slot's Biolink range where bmt declares one (see
# :func:`qualifier_enum_name`); slots whose enum is not on the slot itself fall
# back to _ENUM_FALLBACKS.
ENUM_QUALIFIER_SLOTS: tuple[str, ...] = (
    "subject_aspect_qualifier",
    "object_aspect_qualifier",
    "subject_direction_qualifier",
    "object_direction_qualifier",
    "subject_derivative_qualifier",
    "object_derivative_qualifier",
    "causal_mechanism_qualifier",
)

# Free-text qualifier slots (no controlled vocabulary).
FREETEXT_QUALIFIER_SLOTS: tuple[str, ...] = (
    "qualified_predicate",
    "subject_part_qualifier",
    "object_part_qualifier",
    "subject_form_or_variant_qualifier",
    "object_form_or_variant_qualifier",
)

# Aspect and derivative qualifiers carry no enum range on the slot itself (it is
# assigned per-association via slot_usage), so bmt cannot derive it; supply it.
_ENUM_FALLBACKS: dict[str, str] = {
    "subject_aspect_qualifier": "GeneOrGeneProductOrChemicalEntityAspectEnum",
    "object_aspect_qualifier": "GeneOrGeneProductOrChemicalEntityAspectEnum",
    "subject_derivative_qualifier": "ChemicalEntityDerivativeEnum",
    "object_derivative_qualifier": "ChemicalEntityDerivativeEnum",
}


@lru_cache(maxsize=1)
def toolkit() -> Toolkit:
    """Return the process-wide ``bmt.Toolkit`` singleton."""
    return Toolkit()


def humanize(token: str) -> str:
    """Turn a CURIE/enum token into readable text.

    Strips any prefix (``biolink:causes`` -> ``causes``) and replaces
    underscores with spaces (``activity_or_abundance`` -> ``activity or
    abundance``).
    """
    local = token.split(":", 1)[-1] if ":" in token else token
    return local.replace("_", " ").strip()


def _local(token: str) -> str:
    """Bare snake_case token: strip a CURIE prefix, keep the underscores."""
    return token.split(":", 1)[-1] if ":" in token else token


def is_valid_predicate(name: str) -> bool:
    """True if ``name`` is a predicate in the Biolink Model."""
    return toolkit().is_predicate(name)


@lru_cache(maxsize=1)
def _all_predicates() -> tuple[str, ...]:
    """Every Biolink predicate as a bare snake_case token (``related to`` root)."""
    descendants = toolkit().get_descendants(
        "related to", reflexive=True, mixin=True, formatted=True
    )
    return tuple(_local(p) for p in descendants)


@lru_cache(maxsize=None)
def candidate_predicates(subject_class: str, object_class: str) -> frozenset[str]:
    """Predicates that can connect ``subject_class`` -> ``object_class``.

    A predicate qualifies when the subject is-a its effective domain *and* the
    object is-a its effective range; a predicate that declares no domain (resp.
    range) is treated as unrestricted on that side. Tokens are bare snake_case
    (``affects``, ``is_substrate_of``) matching the rest of the library. Returns
    an empty set when either class is unknown to bmt, so callers can fall back to
    an unconstrained predicate.

    Note this honours *inherited* domain/range: core predicates like ``affects``
    and ``causes`` declare none of their own (they inherit ``named thing`` from
    ``related to``) and so are correctly admitted — unlike
    ``Toolkit.get_all_predicates_with_class_domain``, which keys off a predicate's
    own ``domain`` and silently drops them. The constraint is therefore permissive
    and does not enforce edge direction: a predicate and its non-canonical inverse
    can both appear, which is intentional — text may state a relation in either
    order while the subject/object stay fixed.
    """
    tk = toolkit()
    if tk.get_element(subject_class) is None or tk.get_element(object_class) is None:
        return frozenset()
    subject_ancestors = set(tk.get_ancestors(subject_class, reflexive=True, mixin=True))
    object_ancestors = set(tk.get_ancestors(object_class, reflexive=True, mixin=True))

    candidates = []
    for predicate in _all_predicates():
        name = humanize(predicate)
        domain = tk.get_slot_domain(name, include_ancestors=False, mixin=True)
        range_ = tk.get_slot_range(name, include_ancestors=False, mixin=True)
        subject_ok = not domain or any(d in subject_ancestors for d in domain)
        object_ok = not range_ or any(r in object_ancestors for r in range_)
        if subject_ok and object_ok:
            candidates.append(predicate)
    return frozenset(candidates)


@lru_cache(maxsize=None)
def predicate_definition(predicate: str) -> str | None:
    """The Biolink definition for a ``predicate``, whitespace-normalized."""
    element = toolkit().get_element(humanize(predicate))
    if element is None or not element.description:
        return None
    return " ".join(element.description.split())


@lru_cache(maxsize=None)
def predicate_tree(candidates: frozenset[str]) -> tuple[tuple[int, str, bool], ...]:
    """``(depth, predicate, is_leaf)`` rows for the candidates' ``is_a`` subtree.

    Rows are in tree (pre-order) order; ``depth`` counts only candidate ancestors
    so the rendered tree is compact, and ``is_leaf`` marks candidates with no more
    specific candidate beneath them (the most specific choices). Conveys predicate
    specificity to the model: deeper ``is_a`` descendants are more specific.
    """
    tk = toolkit()
    rows: list[tuple[int, str, bool]] = []

    def walk(name: str, depth: int) -> None:
        token = _local(name).replace(" ", "_")
        is_candidate = token in candidates
        if is_candidate:
            descendants = tk.get_descendants(
                name, reflexive=False, mixin=True, formatted=True
            )
            is_leaf = not any(_local(d) in candidates for d in descendants)
            rows.append((depth, token, is_leaf))
        for child in sorted(tk.get_children(name, mixin=True)):
            walk(child, depth + 1 if is_candidate else depth)

    walk("related to", 0)
    return tuple(rows)


@lru_cache(maxsize=None)
def qualifier_enum_name(slot: str) -> str | None:
    """Backing enum name for an enum-valued qualifier ``slot``, else ``None``.

    Read from the slot's Biolink ``range`` when that resolves to an enum;
    otherwise from :data:`_ENUM_FALLBACKS`. Free-text slots return ``None``.
    """
    element = toolkit().get_element(humanize(slot))
    if element is not None:
        range_ = getattr(element, "range", None)
        if range_ and toolkit().view.get_enum(range_) is not None:
            return range_
    return _ENUM_FALLBACKS.get(slot)


@lru_cache(maxsize=None)
def qualifier_description(slot: str) -> str | None:
    """The Biolink ``SlotDefinition`` description for a qualifier ``slot``.

    Whitespace-normalized; ``None`` if the slot is unknown or undocumented.
    """
    element = toolkit().get_element(humanize(slot))
    if element is None or not element.description:
        return None
    return " ".join(element.description.split())


@lru_cache(maxsize=None)
def permissible_values(enum_name: str) -> frozenset[str]:
    """Permissible value tokens for a Biolink enum, empty if unknown."""
    enum = toolkit().view.get_enum(enum_name)
    if enum is None:
        return frozenset()
    return frozenset(enum.permissible_values.keys())


def validate_qualifier(slot: str, value: str) -> bool:
    """True if ``value`` is permissible for an enum-valued qualifier ``slot``.

    Slots without a known backing enum (e.g. free-text part qualifiers) always
    validate as ``True``.
    """
    enum_name = qualifier_enum_name(slot)
    if enum_name is None:
        return True
    return value in permissible_values(enum_name)
