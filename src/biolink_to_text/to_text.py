"""Serialize a :class:`QualifiedStatement` into a natural-language sentence.

Composition follows the Biolink Model's strict qualifier ordering described in
https://biolink.github.io/biolink-model/reading-a-qualifier-based-statement/ :
each qualifier layer wraps all the preceding ones.
"""

from __future__ import annotations

from biolink_to_text.model import Entity, Path, PathGroup, QualifiedStatement
from biolink_to_text.schema import humanize

# Predicate -> verb phrase. Anything not listed falls back to humanize().
_PREDICATE_VERBS: dict[str, str] = {
    "causes": "causes",
    "contributes_to": "contributes to",
    "affects": "affects",
    "regulates": "regulates",
    "physically_interacts_with": "physically interacts with",
    "directly_physically_interacts_with": "binds",
    "related_to": "is related to",
}

# Causal mechanism -> verb phrase, used when no explicit predicate verb fits.
_MECHANISM_VERBS: dict[str, str] = {
    "agonism": "is an agonist of",
    "inverse_agonism": "is an inverse agonist of",
    "antagonism": "is an antagonist of",
    "inhibition": "inhibits",
    "activation": "activates",
    "binding": "binds",
}


def _context_label(ctx: Entity | str | None) -> str | None:
    if ctx is None:
        return None
    return ctx.display if isinstance(ctx, Entity) else humanize(ctx)


def _entity_phrase(
    label: str,
    *,
    part: str | None,
    form_or_variant: str | None,
    aspect: str | None,
    direction: str | None,
    derivative: str | None,
) -> str:
    """Compose one side's noun phrase from ``label``, inside-out, per Biolink ordering.

    ``label`` is the entity's surface form (``Entity.display``) for :func:`to_text`,
    or a placeholder such as ``"subject"`` for :func:`to_anonymized_text`.
    """
    phrase = label

    if derivative:
        phrase = f"{phrase} {humanize(derivative)}"
    if part:
        phrase = f"{phrase} {humanize(part)}"
    if form_or_variant:
        phrase = f"a {humanize(form_or_variant)} form of the {phrase}"
    if aspect:
        phrase = f"{humanize(aspect)} of {phrase}"
    if direction:
        phrase = f"{humanize(direction)} {phrase}"
    return phrase


def _local(token: str) -> str:
    """Strip a CURIE prefix but keep the underscored Biolink token."""
    return token.split(":", 1)[-1] if ":" in token else token


def _predicate_phrase(stmt: QualifiedStatement) -> str:
    if stmt.qualified_predicate:
        return humanize(stmt.qualified_predicate)
    # A causal mechanism gives a more specific verb than a generic predicate.
    if stmt.causal_mechanism_qualifier in _MECHANISM_VERBS:
        return _MECHANISM_VERBS[stmt.causal_mechanism_qualifier]
    pred = _local(stmt.predicate)
    if pred in _PREDICATE_VERBS:
        return _PREDICATE_VERBS[pred]
    return humanize(pred)


def to_text(stmt: QualifiedStatement, anonymize_entities: bool = False) -> str:
    """Render a qualified Biolink statement as an English sentence.

    With ``anonymize_entities=True``, the subject, object, and context nodes are
    replaced with generic placeholders while the predicate and every qualifier
    are kept intact. For example, the statement rendered as
    ``"Chemical A causes increased expression of Gene B"`` becomes
    ``"subject causes increased expression of object"``. Useful for building
    training data that captures predicate/qualifier patterns without leaking the
    specific entities involved.
    """
    if anonymize_entities:
        subject_label, object_label = "subject", "object"
        anatomical_context = (
            "an anatomical context" if stmt.anatomical_context_qualifier else None
        )
        species_context = (
            "a species context" if stmt.species_context_qualifier else None
        )
    else:
        subject_label = stmt.subject.display
        object_label = stmt.object.display
        anatomical_context = _context_label(stmt.anatomical_context_qualifier)
        species_context = _context_label(stmt.species_context_qualifier)

    subject = _entity_phrase(
        subject_label,
        part=stmt.subject_part_qualifier,
        form_or_variant=stmt.subject_form_or_variant_qualifier,
        aspect=stmt.subject_aspect_qualifier,
        direction=stmt.subject_direction_qualifier,
        derivative=stmt.subject_derivative_qualifier,
    )
    obj = _entity_phrase(
        object_label,
        part=stmt.object_part_qualifier,
        form_or_variant=stmt.object_form_or_variant_qualifier,
        aspect=stmt.object_aspect_qualifier,
        direction=stmt.object_direction_qualifier,
        derivative=stmt.object_derivative_qualifier,
    )

    sentence = f"{subject} {_predicate_phrase(stmt)} {obj}"

    for ctx in (anatomical_context, species_context):
        if ctx:
            sentence = f"{sentence} in {ctx}"

    return sentence


def path_to_text(path: Path, anonymize_entities: bool = False) -> list[str]:
    """Render a path as its list of triple sentences, in path order.

    Pass ``anonymize_entities=True`` to emit node-free sentences (see
    :func:`to_text`).
    """
    return [to_text(statement, anonymize_entities) for statement in path.statements]


def paths_to_text(
    paths: list[Path] | list[PathGroup],
    anonymize_entities: bool = False,
) -> list[list[str]] | list[list[list[str]]]:
    """Render many paths to sentences, preserving their grouping.

    Accepts the output of :func:`biolink_to_text.from_trapi.paths_from_trapi` in
    either form. A ``list[Path]`` becomes ``list[list[str]]`` (one sentence list
    per path); a ``list[PathGroup]`` becomes ``list[list[list[str]]]``, keeping
    each result's paths nested together.

    Pass ``anonymize_entities=True`` to emit node-free sentences (see
    :func:`to_text`).
    """
    return [
        [path_to_text(path, anonymize_entities) for path in item.paths]
        if isinstance(item, PathGroup)
        else path_to_text(item, anonymize_entities)
        for item in paths
    ]
