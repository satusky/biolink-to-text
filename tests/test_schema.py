"""Tests for the bmt-backed schema helpers."""

from biolink_to_text.schema import (
    ENUM_QUALIFIER_SLOTS,
    candidate_predicates,
    predicate_definition,
    predicate_tree,
    qualifier_description,
    qualifier_enum_name,
)


def test_candidate_predicates_is_directional_intersection():
    cands = candidate_predicates("chemical entity", "gene")
    # Tokens are bare snake_case, matching the rest of the library.
    assert all(":" not in p and p == p.lower() for p in cands)
    # A relation that fits chemical -> gene is present...
    assert "affects" in cands
    # ...while one that cannot (gene/disease domain) is absent.
    assert "has_phenotype" not in cands


def test_candidate_predicates_includes_inherited_domain_predicates():
    """Core predicates that inherit (not declare) their domain must survive."""
    cands = candidate_predicates("chemical entity", "gene")
    assert {"affects", "causes", "regulates", "contributes_to"} <= cands


def test_candidate_predicates_keeps_both_directional_forms():
    """Non-canonical inverses stay available so either surface order works."""
    cands = candidate_predicates("small molecule", "protein")
    assert "is_substrate_of" in cands and "has_substrate" in cands


def test_predicate_tree_orders_by_specificity_and_marks_leaves():
    cands = candidate_predicates("chemical entity", "gene")
    rows = predicate_tree(cands)
    depth = {token: d for d, token, _ in rows}
    leaf = {token: is_leaf for _, token, is_leaf in rows}
    # Every candidate appears exactly once.
    assert {token for _, token, _ in rows} == cands
    # Specificity is reflected by depth: causes is deeper than its parent.
    assert depth["causes"] > depth["contributes_to"] > depth["related_to"]
    # A predicate with a more specific candidate beneath it is not a leaf...
    assert leaf["causes"] and not leaf["affects"]  # regulates/disrupts under affects


def test_predicate_definition():
    assert predicate_definition("causes")
    assert predicate_definition("not_a_predicate") is None


def test_candidate_predicates_unknown_class_is_empty():
    assert candidate_predicates("not a real class", "gene") == frozenset()


def test_qualifier_enum_name_derived_and_fallback():
    # Derived from the slot's Biolink range.
    assert qualifier_enum_name("object_direction_qualifier") == "DirectionQualifierEnum"
    assert (
        qualifier_enum_name("causal_mechanism_qualifier")
        == "CausalMechanismQualifierEnum"
    )
    # Falls back where bmt declares no enum range on the slot itself.
    assert (
        qualifier_enum_name("object_aspect_qualifier")
        == "GeneOrGeneProductOrChemicalEntityAspectEnum"
    )
    assert (
        qualifier_enum_name("subject_derivative_qualifier")
        == "ChemicalEntityDerivativeEnum"
    )


def test_every_enum_slot_resolves_to_an_enum():
    assert all(qualifier_enum_name(slot) for slot in ENUM_QUALIFIER_SLOTS)


def test_qualifier_description_present_and_freetext():
    assert "aspect" in (qualifier_description("object_aspect_qualifier") or "").lower()
    # Free-text slots are documented too.
    assert qualifier_description("object_part_qualifier")
