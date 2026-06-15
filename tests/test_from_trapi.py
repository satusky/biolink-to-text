"""Tests for building triples and paths from TRAPI messages.

Fixtures are small hand-written TRAPI dicts modelled on the structure of
``example_data/ROBOKOP_answer.json`` (the real ~55 MB file is only exercised by
an optional, guarded smoke test).
"""

import json
import os

import pytest

from biolink_to_text import (
    Entity,
    Path,
    PathGroup,
    QualifiedStatement,
    path_to_text,
    paths_from_trapi,
    paths_to_text,
    statement_from_kg_edge,
    to_text,
    triples_from_trapi,
)

# A CTD-style qualified edge: tobacco smoke increases expression of NCBIGene:5743.
_NODES = {
    "MESH:D014028": {"name": "Tobacco Smoke Pollution"},
    "NCBIGene:5743": {"name": "PTGS2"},
    "UBERON:0001013": {"name": "adipose tissue"},
}

_QUALIFIED_EDGE = {
    "subject": "MESH:D014028",
    "object": "NCBIGene:5743",
    "predicate": "biolink:affects",
    "qualifiers": [
        {
            "qualifier_type_id": "biolink:object_direction_qualifier",
            "qualifier_value": "increased",
        },
        {
            "qualifier_type_id": "biolink:qualified_predicate",
            "qualifier_value": "biolink:causes",
        },
        {
            "qualifier_type_id": "biolink:object_aspect_qualifier",
            "qualifier_value": "expression",
        },
    ],
}


def test_statement_from_qualified_kg_edge():
    stmt = statement_from_kg_edge(_QUALIFIED_EDGE, _NODES)
    assert stmt == QualifiedStatement(
        subject=Entity("MESH:D014028", "Tobacco Smoke Pollution"),
        predicate="affects",
        object=Entity("NCBIGene:5743", "PTGS2"),
        qualified_predicate="biolink:causes",
        object_aspect_qualifier="expression",
        object_direction_qualifier="increased",
    )
    assert to_text(stmt) == (
        "Tobacco Smoke Pollution causes increased expression of PTGS2"
    )


def test_category_populated_from_node_categories():
    nodes = {
        "MESH:D014028": {
            "name": "Tobacco Smoke Pollution",
            "categories": ["biolink:ChemicalEntity", "biolink:NamedThing"],
        },
        "NCBIGene:5743": {"name": "PTGS2"},  # no categories
    }
    stmt = statement_from_kg_edge(_QUALIFIED_EDGE, nodes)
    # First (most specific) category is kept; absent categories stay None.
    assert stmt.subject.category == "biolink:ChemicalEntity"
    assert stmt.object.category is None


def test_label_resolution_falls_back_to_curie():
    edge = {
        "subject": "MESH:D014028",
        "object": "NCBIGene:9999",  # absent from _NODES
        "predicate": "biolink:related_to",
    }
    stmt = statement_from_kg_edge(edge, _NODES)
    assert stmt.subject == Entity("MESH:D014028", "Tobacco Smoke Pollution")
    assert stmt.object == Entity("NCBIGene:9999", None)
    assert to_text(stmt) == "Tobacco Smoke Pollution is related to NCBIGene:9999"


def test_entity_context_qualifier_resolved_to_label():
    edge = {
        "subject": "MESH:D014028",
        "object": "NCBIGene:5743",
        "predicate": "biolink:affects",
        "qualifiers": [
            {
                "qualifier_type_id": "biolink:anatomical_context_qualifier",
                "qualifier_value": "UBERON:0001013",
            },
        ],
    }
    stmt = statement_from_kg_edge(edge, _NODES)
    assert stmt.anatomical_context_qualifier == Entity(
        "UBERON:0001013", "adipose tissue"
    )
    assert to_text(stmt).endswith("in adipose tissue")


def test_unknown_qualifier_is_ignored():
    edge = {
        "subject": "MESH:D014028",
        "object": "NCBIGene:5743",
        "predicate": "biolink:affects",
        "qualifiers": [
            {"qualifier_type_id": "biolink:made_up_qualifier", "qualifier_value": "x"},
        ],
    }
    stmt = statement_from_kg_edge(edge, _NODES)
    assert stmt.qualifiers() == {}


_EDGES = {
    "e_1": _QUALIFIED_EDGE,
    "e_2": {
        "subject": "NCBIGene:5743",
        "object": "MONDO:0004979",
        "predicate": "biolink:related_to",
    },
    "e_3": {
        "subject": "MESH:D014028",
        "object": "MONDO:0004979",
        "predicate": "biolink:contributes_to",
    },
    # Parallel to e_1 (same node pair) but a distinct, unqualified triple.
    "e_4": {
        "subject": "MESH:D014028",
        "object": "NCBIGene:5743",
        "predicate": "biolink:affects",
    },
}
_KG_NODES = {**_NODES, "MONDO:0004979": {"name": "asthma"}}


def _result(score, edge_bindings):
    return {"score": score, "analyses": [{"edge_bindings": edge_bindings}]}


def _message(results):
    return {
        "knowledge_graph": {"nodes": _KG_NODES, "edges": _EDGES},
        "results": results,
    }


def test_path_dedups_edges_within_result_in_order():
    # e_1 bound twice within the result -> one triple, no per-edge count.
    msg = _message([
        _result(0.95, {"e0": [{"id": "e_1"}, {"id": "e_1"}], "e1": [{"id": "e_2"}]})
    ])
    paths = paths_from_trapi(msg)
    assert len(paths) == 1
    path = paths[0]
    assert path.count == 1
    assert path.score == 0.95
    assert path_to_text(path) == [
        "Tobacco Smoke Pollution causes increased expression of PTGS2",
        "PTGS2 is related to asthma",
    ]


def test_identical_paths_across_results_collapse_with_count():
    # Two results yield the same path (same triples, same order), one differs.
    same = {"e0": [{"id": "e_1"}], "e1": [{"id": "e_2"}]}
    msg = _message([
        _result(0.80, same),
        _result(0.91, same),
        _result(0.50, {"e0": [{"id": "e_3"}]}),
    ])
    paths = paths_from_trapi(msg)
    assert len(paths) == 2

    merged, other = paths
    assert merged.count == 2
    assert merged.score == pytest.approx(0.855)  # mean of 0.80 and 0.91
    assert other.count == 1
    assert path_to_text(other) == ["Tobacco Smoke Pollution contributes to asthma"]


def test_triples_from_trapi_one_per_edge():
    triples = triples_from_trapi(_message([]))
    assert len(triples) == 4


def test_enumerate_edges_cartesian_product():
    # e0 binds two distinct parallel triples (e_1, e_4); e1 binds one (e_2).
    msg = _message([
        _result(0.9, {"e0": [{"id": "e_1"}, {"id": "e_4"}], "e1": [{"id": "e_2"}]})
    ])

    single = paths_from_trapi(msg, enumerate_edges=True)
    assert len(single) == 2  # 2 (e0) x 1 (e1)
    assert all(len(p.statements) == 2 for p in single)  # one triple per query edge
    assert all(p.count == 1 and p.score == 0.9 for p in single)

    bundle = paths_from_trapi(msg)  # default: all parallel edges in one path
    assert len(bundle) == 1
    assert len(bundle[0].statements) == 3  # e_1, e_4, e_2


def test_enumerate_dedups_identical_paths_across_results():
    # Two results yield the same single-edge path; a third differs.
    same = {"e0": [{"id": "e_1"}], "e1": [{"id": "e_2"}]}
    msg = _message([
        _result(0.80, same),
        _result(0.91, same),
        _result(0.50, {"e0": [{"id": "e_3"}]}),
    ])
    paths = paths_from_trapi(msg, enumerate_edges=True)
    assert len(paths) == 2

    merged, other = paths
    assert merged.count == 2
    assert merged.score == pytest.approx(0.855)  # mean of 0.80 and 0.91
    assert other.count == 1
    assert path_to_text(other) == ["Tobacco Smoke Pollution contributes to asthma"]


def test_group_by_result_keeps_each_result_together():
    # Result A enumerates to 2 single-edge paths; result B to 1.
    msg = _message([
        _result(0.9, {"e0": [{"id": "e_1"}, {"id": "e_4"}], "e1": [{"id": "e_2"}]}),
        _result(0.4, {"e0": [{"id": "e_3"}]}),
    ])
    groups = paths_from_trapi(msg, enumerate_edges=True, group_by_result=True)

    assert [type(g) for g in groups] == [PathGroup, PathGroup]
    a, b = groups
    assert a.score == 0.9
    assert len(a.paths) == 2  # cartesian product kept together
    assert all(isinstance(p, Path) and len(p.statements) == 2 for p in a.paths)
    assert b.score == 0.4
    assert len(b.paths) == 1
    assert path_to_text(b.paths[0]) == [
        "Tobacco Smoke Pollution contributes to asthma"
    ]


def test_group_by_result_bundle_mode_one_path_per_group():
    msg = _message([
        _result(0.9, {"e0": [{"id": "e_1"}, {"id": "e_4"}], "e1": [{"id": "e_2"}]})
    ])
    groups = paths_from_trapi(msg, group_by_result=True)  # bundle mode
    assert len(groups) == 1
    (group,) = groups
    assert len(group.paths) == 1  # one bundle path
    assert len(group.paths[0].statements) == 3  # e_1, e_4, e_2
    assert group.score == 0.9


def test_group_by_result_skips_resultless_paths():
    # A result whose only bound edge id is unknown produces no paths.
    msg = _message([_result(0.9, {"e0": [{"id": "missing"}]})])
    assert paths_from_trapi(msg, enumerate_edges=True, group_by_result=True) == []


def test_paths_to_text_flat_list_of_paths():
    msg = _message([
        _result(0.9, {"e0": [{"id": "e_1"}, {"id": "e_4"}], "e1": [{"id": "e_2"}]})
    ])
    paths = paths_from_trapi(msg, enumerate_edges=True)  # list[Path]
    rendered = paths_to_text(paths)
    # One sentence list per path; same order as path_to_text per path.
    assert rendered == [path_to_text(p) for p in paths]
    assert rendered[0] == [
        "Tobacco Smoke Pollution causes increased expression of PTGS2",
        "PTGS2 is related to asthma",
    ]


def test_paths_to_text_preserves_group_nesting():
    msg = _message([
        _result(0.9, {"e0": [{"id": "e_1"}, {"id": "e_4"}], "e1": [{"id": "e_2"}]}),
        _result(0.4, {"e0": [{"id": "e_3"}]}),
    ])
    groups = paths_from_trapi(msg, enumerate_edges=True, group_by_result=True)
    rendered = paths_to_text(groups)  # list[PathGroup] -> list[list[list[str]]]

    assert [len(group) for group in rendered] == [2, 1]  # paths per result
    assert rendered == [[path_to_text(p) for p in g.paths] for g in groups]
    assert rendered[1] == [["Tobacco Smoke Pollution contributes to asthma"]]


_REAL_FILE = os.path.join(
    os.path.dirname(__file__), "..", "example_data", "ROBOKOP_answer.json"
)


@pytest.mark.skipif(
    not os.path.exists(_REAL_FILE), reason="example ROBOKOP answer not present"
)
def test_real_robokop_answer_smoke():
    with open(_REAL_FILE) as fh:
        message = json.load(fh)["message"]
    triples = triples_from_trapi(message)
    paths = paths_from_trapi(message)
    assert triples and paths
    assert to_text(triples[0])
    assert all(path.count >= 1 and path.statements for path in paths)
