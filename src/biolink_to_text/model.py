"""Data model for qualified Biolink statements and the paths they form."""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Entity:
    """A KG node: an ontological identifier with an optional human label.

    Ontological mapping is out of scope for this library; callers supply the
    ``curie`` (and ideally a ``label``) from their own entity-resolution tools.

    ``category`` is the node's Biolink class (e.g. ``"biolink:ChemicalEntity"``
    or ``"chemical entity"``), also caller-supplied. When set for both the
    subject and object, :func:`biolink_to_text.from_text.from_text` uses it to
    constrain the candidate predicates offered to the model.
    """

    curie: str
    label: str | None = None
    category: str | None = None

    @property
    def display(self) -> str:
        """Human-readable surface form, falling back to the CURIE."""
        return self.label or self.curie


@dataclass(frozen=True)
class QualifiedStatement:
    """A Biolink (subject, predicate, object) triple plus its qualifiers.

    Qualifier slot names mirror the Biolink Model. Unused slots are ``None``.
    Enum-valued qualifiers (direction, aspect, derivative, causal mechanism)
    hold the permissible-value token, e.g. ``"increased"`` or ``"methylation"``.
    """

    subject: Entity
    predicate: str
    object: Entity

    qualified_predicate: str | None = None

    # Subject-side qualifiers
    subject_aspect_qualifier: str | None = None
    subject_direction_qualifier: str | None = None
    subject_part_qualifier: str | None = None
    subject_form_or_variant_qualifier: str | None = None
    subject_derivative_qualifier: str | None = None

    # Object-side qualifiers
    object_aspect_qualifier: str | None = None
    object_direction_qualifier: str | None = None
    object_part_qualifier: str | None = None
    object_form_or_variant_qualifier: str | None = None
    object_derivative_qualifier: str | None = None

    # Statement-level context qualifiers
    causal_mechanism_qualifier: str | None = None
    anatomical_context_qualifier: Entity | str | None = None
    species_context_qualifier: Entity | str | None = None

    def qualifiers(self) -> dict[str, object]:
        """Return the populated qualifier slots as a ``{name: value}`` dict."""
        skip = {"subject", "predicate", "object"}
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in skip and getattr(self, f.name) is not None
        }


@dataclass(frozen=True)
class Path:
    """An ordered sequence of unique triples derived from query results.

    A path corresponds to one query answer. ``count`` records how many results
    produced this exact path (its triples, in order), for result weighting;
    ``score`` is the representative (highest) result score among them.

    For now a path is simply a list of triples; the multi-hop chaining of those
    triples is left for a later iteration.
    """

    statements: tuple[QualifiedStatement, ...]
    count: int = 1
    score: float | None = None


@dataclass(frozen=True)
class PathGroup:
    """All paths derived from a single query result, kept together.

    Produced when paths are grouped by result (see
    :func:`biolink_to_text.from_trapi.paths_from_trapi`): the result's enumerated
    single-edge paths are unique within the group, so they carry no per-path
    count; the answer-level ``score`` weights the group as a whole.
    """

    paths: tuple[Path, ...]
    score: float | None = None
