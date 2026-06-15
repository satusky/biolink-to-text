"""Parse natural-language statements into qualified Biolink triples via an LLM.

Entity recognition and ontological CURIE mapping are out of scope: the caller
passes the already-resolved ``subject`` and ``object`` entities. This module
extracts only the *predicate* and *qualifiers* connecting them, using a
provider-agnostic :class:`LLMClient`, and validates the result against ``bmt``.

When the subject and object carry a Biolink ``category``, the predicate is
constrained to those that can legally connect the two classes (see
:func:`biolink_to_text.schema.candidate_predicates`); each qualifier slot is
annotated with its Biolink description to give the model context.
"""

from __future__ import annotations

from biolink_to_text.llm import LLMClient
from biolink_to_text.model import Entity, QualifiedStatement
from biolink_to_text.schema import (
    ENUM_QUALIFIER_SLOTS,
    FREETEXT_QUALIFIER_SLOTS,
    candidate_predicates,
    is_valid_predicate,
    permissible_values,
    qualifier_description,
    qualifier_enum_name,
    validate_qualifier,
)

_SYSTEM = (
    "You convert a biomedical statement into a Biolink Model qualified edge. "
    "The subject and object entities are already identified for you. Determine "
    "the predicate and any qualifiers that hold between them. Only use values "
    "from the provided vocabularies. Omit qualifiers that do not apply."
)


def _candidate_predicates(subject: Entity, object: Entity) -> frozenset[str]:
    """Predicates allowed between the entities' classes, empty if unknown."""
    if subject.category and object.category:
        return candidate_predicates(subject.category, object.category)
    return frozenset()


def _response_schema(subject: Entity, object: Entity) -> dict:
    """JSON schema constraining the model to the supported qualifier slots.

    When both entities have a Biolink class, the ``predicate`` is restricted to
    the candidate set; otherwise it is left as a free Biolink predicate token.
    """
    predicate: dict = {"type": "string", "description": "Biolink predicate token"}
    candidates = _candidate_predicates(subject, object)
    if candidates:
        predicate["enum"] = sorted(candidates)

    properties: dict[str, dict] = {"predicate": predicate}
    for slot in FREETEXT_QUALIFIER_SLOTS:
        properties[slot] = {"type": ["string", "null"]}
    for slot in ENUM_QUALIFIER_SLOTS:
        properties[slot] = {
            "type": ["string", "null"],
            "enum": [*sorted(permissible_values(qualifier_enum_name(slot))), None],
        }
    return {
        "type": "object",
        "properties": properties,
        "required": ["predicate"],
        "additionalProperties": False,
    }


def _entity_line(role: str, entity: Entity) -> str:
    line = f"{role}: {entity.display} ({entity.curie})"
    return f"{line} [{entity.category}]" if entity.category else line


def _qualifier_line(slot: str) -> str:
    """One prompt bullet: the slot, its Biolink description, and any vocabulary."""
    line = f"- {slot}"
    description = qualifier_description(slot)
    if description:
        line += f" — {description}"
    enum_name = qualifier_enum_name(slot)
    if enum_name:
        values = ", ".join(sorted(permissible_values(enum_name)))
        line += f"\n  Allowed values: {values}"
    return line


def _build_prompt(text: str, subject: Entity, object: Entity) -> str:
    sections = [
        f"Statement: {text!r}",
        _entity_line("Subject", subject),
        _entity_line("Object", object),
    ]

    candidates = _candidate_predicates(subject, object)
    if candidates:
        sections.append(
            "Candidate predicates (choose exactly one):\n  "
            + ", ".join(sorted(candidates))
        )

    qualifier_lines = [
        _qualifier_line(slot)
        for slot in (*ENUM_QUALIFIER_SLOTS, *FREETEXT_QUALIFIER_SLOTS)
    ]
    sections.append(
        "Qualifiers (omit any that do not apply):\n" + "\n".join(qualifier_lines)
    )
    return "\n\n".join(sections)


def from_text(
    text: str,
    llm: LLMClient,
    *,
    subject: Entity,
    object: Entity,
) -> QualifiedStatement:
    """Extract a :class:`QualifiedStatement` from ``text``.

    ``subject`` and ``object`` are caller-provided, already-mapped entities. If
    both carry a Biolink ``category``, the predicate is constrained to those
    valid between the two classes. The returned statement's predicate and enum
    qualifiers are validated against the Biolink Model; invalid values raise
    :class:`ValueError`.
    """
    result = llm.complete(
        system=_SYSTEM,
        prompt=_build_prompt(text, subject, object),
        response_schema=_response_schema(subject, object),
    )

    predicate = result.get("predicate")
    if not predicate or not is_valid_predicate(predicate):
        raise ValueError(f"LLM returned a non-Biolink predicate: {predicate!r}")
    candidates = _candidate_predicates(subject, object)
    if candidates and predicate not in candidates:
        raise ValueError(
            f"Predicate {predicate!r} cannot connect the given "
            f"{subject.category} -> {object.category}"
        )

    kwargs: dict[str, object] = {}
    for slot in (*ENUM_QUALIFIER_SLOTS, *FREETEXT_QUALIFIER_SLOTS):
        value = result.get(slot)
        if value is None:
            continue
        if not validate_qualifier(slot, value):
            raise ValueError(f"Invalid value {value!r} for qualifier {slot!r}")
        kwargs[slot] = value

    return QualifiedStatement(
        subject=subject,
        predicate=predicate,
        object=object,
        **kwargs,
    )
