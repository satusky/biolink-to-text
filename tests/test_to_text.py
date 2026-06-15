"""Golden tests for triple -> text, based on the documented Biolink examples.

Source statements:
https://biolink.github.io/biolink-model/reading-a-qualifier-based-statement/
https://biolink.github.io/biolink-model/association-examples-with-qualifiers/

Where the documentation uses idiomatic lexicalizations (e.g. "deficiency" for
decreased abundance), the expected string here is the *compositional* reading
the generator produces, which is semantically equivalent.
"""

from biolink_to_text import Entity, QualifiedStatement, to_text


def test_canonical_fully_qualified_cdkn2a():
    """The headline example from the Biolink 'reading a qualified edge' page."""
    stmt = QualifiedStatement(
        subject=Entity("CHEBI:5868", "Hexachlorobenzene"),
        predicate="affects",
        qualified_predicate="biolink:causes",
        object=Entity("HGNC:1787", "CDKN2A"),
        subject_derivative_qualifier="metabolite",
        object_part_qualifier="promoter",
        object_form_or_variant_qualifier="mutant",
        object_aspect_qualifier="methylation",
        object_direction_qualifier="increased",
        anatomical_context_qualifier="the nucleus of HeLa cells",
    )
    assert to_text(stmt) == (
        "Hexachlorobenzene metabolite causes increased methylation of a mutant "
        "form of the CDKN2A promoter in the nucleus of HeLa cells"
    )


def test_simple_interaction_no_qualifiers():
    stmt = QualifiedStatement(
        subject=Entity("CHEBI:5001", "Fenofibrate"),
        predicate="directly_physically_interacts_with",
        object=Entity("HGNC:9232", "PPARA"),
    )
    assert to_text(stmt) == "Fenofibrate binds PPARA"


def test_object_aspect_without_direction():
    stmt = QualifiedStatement(
        subject=Entity("CHEBI:4026", "Cyclophosphamide"),
        predicate="affects",
        object=Entity("HGNC:2615", "CYP2B6"),
        object_aspect_qualifier="hydroxylation",
    )
    assert to_text(stmt) == "Cyclophosphamide affects hydroxylation of CYP2B6"


def test_object_aspect_and_direction_with_qualified_predicate():
    stmt = QualifiedStatement(
        subject=Entity("CHEBI:33216", "Bisphenol A"),
        predicate="affects",
        qualified_predicate="biolink:causes",
        object=Entity("HGNC:3467", "ESR1"),
        object_aspect_qualifier="degradation",
        object_direction_qualifier="decreased",
    )
    assert to_text(stmt) == "Bisphenol A causes decreased degradation of ESR1"


def test_object_expression_increase_with_anatomical_context():
    stmt = QualifiedStatement(
        subject=Entity("CHEBI:16811", "methionine"),
        predicate="affects",
        object=Entity("HGNC:286", "ADRB2"),
        object_aspect_qualifier="expression",
        object_direction_qualifier="increased",
        anatomical_context_qualifier=Entity("UBERON:0001013", "adipose tissue"),
    )
    assert to_text(stmt) == (
        "methionine affects increased expression of ADRB2 in adipose tissue"
    )


def test_causal_mechanism_agonism():
    stmt = QualifiedStatement(
        subject=Entity("CHEBI:5001", "Fenofibrate"),
        predicate="affects",
        object=Entity("HGNC:9232", "PPARA"),
        causal_mechanism_qualifier="agonism",
    )
    assert to_text(stmt) == "Fenofibrate is an agonist of PPARA"


def test_entity_falls_back_to_curie_without_label():
    stmt = QualifiedStatement(
        subject=Entity("CHEBI:5001"),
        predicate="related_to",
        object=Entity("HGNC:9232"),
    )
    assert to_text(stmt) == "CHEBI:5001 is related to HGNC:9232"


def test_anonymized_strips_nodes_keeps_qualifiers():
    """Same statement as the canonical example, with nodes replaced."""
    stmt = QualifiedStatement(
        subject=Entity("CHEBI:5868", "Hexachlorobenzene"),
        predicate="affects",
        qualified_predicate="biolink:causes",
        object=Entity("HGNC:1787", "CDKN2A"),
        subject_derivative_qualifier="metabolite",
        object_part_qualifier="promoter",
        object_form_or_variant_qualifier="mutant",
        object_aspect_qualifier="methylation",
        object_direction_qualifier="increased",
        anatomical_context_qualifier="the nucleus of HeLa cells",
    )
    assert to_text(stmt, anonymize_entities=True) == (
        "subject metabolite causes increased methylation of a mutant "
        "form of the object promoter in an anatomical context"
    )


def test_anonymized_matches_motivating_example():
    stmt = QualifiedStatement(
        subject=Entity("CHEBI:1", "Chemical A"),
        predicate="affects",
        qualified_predicate="biolink:causes",
        object=Entity("HGNC:1", "Gene B"),
        object_aspect_qualifier="expression",
        object_direction_qualifier="increased",
    )
    assert (
        to_text(stmt, anonymize_entities=True)
        == "subject causes increased expression of object"
    )


def test_path_to_text_accepts_anonymize_flag():
    from biolink_to_text import Path, path_to_text

    stmt = QualifiedStatement(
        subject=Entity("CHEBI:5001", "Fenofibrate"),
        predicate="directly_physically_interacts_with",
        object=Entity("HGNC:9232", "PPARA"),
    )
    path = Path(statements=(stmt,))
    assert path_to_text(path) == ["Fenofibrate binds PPARA"]
    assert path_to_text(path, anonymize_entities=True) == ["subject binds object"]
