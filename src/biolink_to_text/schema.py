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


def is_valid_predicate(name: str) -> bool:
    """True if ``name`` is a predicate in the Biolink Model."""
    return toolkit().is_predicate(name)


@lru_cache(maxsize=None)
def candidate_predicates(subject_class: str, object_class: str) -> frozenset[str]:
    """Predicates that can connect ``subject_class`` -> ``object_class``.

    The intersection of predicates whose domain admits the subject class and
    whose range admits the object class (ancestors and mixins included), as bare
    snake_case tokens (``affects``, ``is_substrate_of``) matching the form used
    elsewhere in the library. Returns an empty set when either class is unknown
    to bmt, so callers can fall back to an unconstrained predicate.

    The constraint is permissive (Biolink domain/range are loose), so it does
    not enforce edge direction: both a predicate and its non-canonical inverse
    can appear, which is intentional — text may state a relation in either order
    while the subject/object stay fixed.
    """
    tk = toolkit()
    if tk.get_element(subject_class) is None or tk.get_element(object_class) is None:
        return frozenset()
    domain = tk.get_all_predicates_with_class_domain(
        subject_class, check_ancestors=True, mixin=True, formatted=True
    )
    range_ = tk.get_all_predicates_with_class_range(
        object_class, check_ancestors=True, mixin=True, formatted=True
    )
    return frozenset(p.split(":", 1)[-1] for p in set(domain) & set(range_))


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
