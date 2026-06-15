"""Build qualified Biolink triples and paths from a TRAPI message.

TRAPI (the Translator Reasoner API) is the response format used by ROBOKOP and
the wider NCATS Translator ecosystem. Facts live in
``message.knowledge_graph.edges``; answers are ``message.results`` that bind a
``query_graph`` onto those edges.

A knowledge-graph edge maps almost 1:1 onto :class:`QualifiedStatement`: its
``subject``/``object`` CURIEs resolve (via ``knowledge_graph.nodes``) to entity
labels, ``predicate`` becomes the predicate, and the ``qualifiers`` list maps
onto our qualifier slots once the ``biolink:`` prefix is stripped.

The input is treated as trusted KG output, so values are not validated against
the Biolink Model here; unrecognized qualifier slots are silently skipped.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import fields

from biolink_to_text.model import Entity, Path, PathGroup, QualifiedStatement

# Qualifier slots on QualifiedStatement that hold an entity/context, which we
# resolve to an Entity (with label) when the value is a known KG node CURIE.
_CONTEXT_SLOTS = frozenset(
    {"anatomical_context_qualifier", "species_context_qualifier"}
)

# Slots settable from TRAPI qualifiers (everything except the core triple).
_QUALIFIER_SLOTS = frozenset(
    f.name
    for f in fields(QualifiedStatement)
    if f.name not in {"subject", "predicate", "object"}
)


def _strip_prefix(token: str) -> str:
    """Drop a CURIE prefix (``biolink:object_direction_qualifier`` -> tail)."""
    return token.split(":", 1)[-1] if ":" in token else token


def _entity(curie: str, nodes: dict) -> Entity:
    """Resolve a CURIE to an :class:`Entity`, labelling it from ``nodes``.

    The node's first (most specific) Biolink ``categories`` entry becomes the
    entity ``category``; absent or empty, it stays ``None``.
    """
    node = nodes.get(curie, {})
    categories = node.get("categories") or []
    return Entity(curie, node.get("name"), categories[0] if categories else None)


def statement_from_kg_edge(edge: dict, nodes: dict) -> QualifiedStatement:
    """Convert one TRAPI knowledge-graph edge into a :class:`QualifiedStatement`.

    ``nodes`` is ``message.knowledge_graph.nodes``, used to label the subject,
    object, and any entity-valued context qualifiers.
    """
    kwargs: dict[str, object] = {}
    for qualifier in edge.get("qualifiers", []) or []:
        slot = _strip_prefix(qualifier["qualifier_type_id"])
        if slot not in _QUALIFIER_SLOTS:
            continue
        value = qualifier["qualifier_value"]
        if slot == "qualified_predicate":
            # Kept verbatim (e.g. "biolink:causes"); to_text humanizes it.
            kwargs[slot] = value
        elif slot in _CONTEXT_SLOTS and value in nodes:
            kwargs[slot] = _entity(value, nodes)
        else:
            kwargs[slot] = value

    return QualifiedStatement(
        subject=_entity(edge["subject"], nodes),
        predicate=_strip_prefix(edge["predicate"]),
        object=_entity(edge["object"], nodes),
        **kwargs,
    )


def triples_from_trapi(message: dict) -> list[QualifiedStatement]:
    """Every knowledge-graph edge as a triple, in the message's edge order."""
    kg = message.get("knowledge_graph", {})
    nodes = kg.get("nodes", {})
    return [
        statement_from_kg_edge(edge, nodes)
        for edge in kg.get("edges", {}).values()
    ]


def _result_edge_ids(result: dict) -> list[str]:
    """Bound KG edge ids for a result, in query-edge order, across analyses."""
    edge_ids: list[str] = []
    for analysis in result.get("analyses", []) or []:
        for bindings in analysis.get("edge_bindings", {}).values():
            edge_ids.extend(binding["id"] for binding in bindings)
    return edge_ids


def _result_statements(result: dict, edges: dict, nodes: dict) -> tuple[QualifiedStatement, ...]:
    """A result's bound triples, flattened and de-duplicated in query-edge order."""
    seen: dict[QualifiedStatement, None] = {}
    for edge_id in _result_edge_ids(result):
        edge = edges.get(edge_id)
        if edge is None:
            continue
        seen[statement_from_kg_edge(edge, nodes)] = None
    return tuple(seen)


def _result_edge_groups(
    result: dict, edges: dict, nodes: dict
) -> list[list[QualifiedStatement]]:
    """A result's triples grouped by query edge, each group de-duplicated.

    Groups are in query-edge (first-seen) order, merged across analyses; missing
    edge ids are skipped and fully-empty groups dropped. This is the grouped
    counterpart to :func:`_result_edge_ids`, used to enumerate single-edge paths.
    """
    groups: dict[str, dict[QualifiedStatement, None]] = {}
    for analysis in result.get("analyses", []) or []:
        for query_edge, bindings in analysis.get("edge_bindings", {}).items():
            group = groups.setdefault(query_edge, {})
            for binding in bindings:
                edge = edges.get(binding["id"])
                if edge is None:
                    continue
                group[statement_from_kg_edge(edge, nodes)] = None
    return [list(group) for group in groups.values() if group]


def _result_paths(
    result: dict, edges: dict, nodes: dict, *, enumerate_edges: bool
) -> tuple[list[tuple[QualifiedStatement, ...]], float | None]:
    """The path(s) a single result contributes, plus its score.

    With ``enumerate_edges`` the result yields one statement tuple per single-edge
    combination (cartesian product across its query edges); otherwise it yields a
    single bundle tuple of all its de-duplicated triples.
    """
    score = result.get("score")
    if enumerate_edges:
        groups = _result_edge_groups(result, edges, nodes)
        tuples = [tuple(combo) for combo in itertools.product(*groups)] if groups else []
    else:
        tuples = [_result_statements(result, edges, nodes)]
    return tuples, score


def _collapse_paths(
    path_scores: Iterable[tuple[tuple[QualifiedStatement, ...], float | None]],
) -> list[Path]:
    """Collapse identical paths, counting occurrences and averaging scores.

    Takes ``(statements, score)`` pairs and returns one :class:`Path` per unique
    statement tuple, in first-seen order, with ``count`` = number of pairs and
    ``score`` = mean of the non-``None`` scores among them.
    """
    counts: dict[tuple[QualifiedStatement, ...], int] = {}
    score_sums: dict[tuple[QualifiedStatement, ...], list[float]] = {}
    for key, score in path_scores:
        counts[key] = counts.get(key, 0) + 1
        if score is not None:
            score_sums.setdefault(key, []).append(score)

    return [
        Path(
            statements=key,
            count=count,
            score=sum(s) / len(s) if (s := score_sums.get(key)) else None,
        )
        for key, count in counts.items()
    ]


def paths_from_trapi(
    message: dict,
    *,
    enumerate_edges: bool = False,
    group_by_result: bool = False,
) -> list[Path] | list[PathGroup]:
    """Query-answer paths from a TRAPI message.

    With ``enumerate_edges=False`` (default) each result yields a single *bundle*
    path: every parallel knowledge-graph edge bound to the result, flattened and
    de-duplicated into one tuple of triples. With ``enumerate_edges=True`` each
    result instead yields the cartesian product across its query edges — one
    triple per query edge — so every distinct single-edge path is returned
    separately. Note enumeration can blow up combinatorially when query edges
    bind many parallel edges.

    With ``group_by_result=False`` (default) a flat ``list[Path]`` is returned:
    identical paths across results collapse into one :class:`Path` whose
    ``count`` records how many results produced it and whose ``score`` is the
    mean result score among them, in first-seen order.

    With ``group_by_result=True`` a ``list[PathGroup]`` is returned instead — one
    :class:`PathGroup` per result (in result order, results yielding no paths
    skipped), keeping that answer's paths together and carrying the result's
    ``score``. No cross-result de-duplication is applied.
    """
    kg = message.get("knowledge_graph", {})
    nodes = kg.get("nodes", {})
    edges = kg.get("edges", {})
    results = message.get("results", []) or []

    if group_by_result:
        groups: list[PathGroup] = []
        for result in results:
            tuples, score = _result_paths(
                result, edges, nodes, enumerate_edges=enumerate_edges
            )
            if not tuples:
                continue
            groups.append(
                PathGroup(
                    paths=tuple(Path(statements=t) for t in tuples),
                    score=score,
                )
            )
        return groups

    def path_scores() -> Iterable[tuple[tuple[QualifiedStatement, ...], float | None]]:
        for result in results:
            tuples, score = _result_paths(
                result, edges, nodes, enumerate_edges=enumerate_edges
            )
            for statements in tuples:
                yield statements, score

    return _collapse_paths(path_scores())
