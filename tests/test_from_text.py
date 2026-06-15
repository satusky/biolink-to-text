"""Tests for text -> triple using a stub LLMClient (no network)."""

import pytest

from biolink_to_text import Entity, QualifiedStatement, from_text, to_text


class StubLLM:
    """An LLMClient that returns a canned structured response."""

    def __init__(self, response: dict):
        self.response = response
        self.last_call: dict | None = None

    def complete(self, *, system: str, prompt: str, response_schema: dict) -> dict:
        self.last_call = {"system": system, "prompt": prompt, "schema": response_schema}
        return self.response


def test_extracts_predicate_and_qualifiers():
    llm = StubLLM(
        {
            "predicate": "affects",
            "qualified_predicate": "biolink:causes",
            "object_aspect_qualifier": "degradation",
            "object_direction_qualifier": "decreased",
        }
    )
    stmt = from_text(
        "Bisphenol A causes decreased degradation of ESR1",
        llm,
        subject=Entity("CHEBI:33216", "Bisphenol A"),
        object=Entity("HGNC:3467", "ESR1"),
    )
    assert stmt.predicate == "affects"
    assert stmt.qualified_predicate == "biolink:causes"
    assert stmt.object_aspect_qualifier == "degradation"
    assert stmt.object_direction_qualifier == "decreased"


def test_prompt_includes_entities_and_vocabulary():
    llm = StubLLM({"predicate": "affects"})
    from_text(
        "methionine affects increased expression of ADRB2",
        llm,
        subject=Entity("CHEBI:16811", "methionine"),
        object=Entity("HGNC:286", "ADRB2"),
    )
    prompt = llm.last_call["prompt"]
    assert "methionine" in prompt and "ADRB2" in prompt
    assert "increased" in prompt  # DirectionQualifierEnum value surfaced


def test_rejects_non_biolink_predicate():
    llm = StubLLM({"predicate": "not_a_real_predicate"})
    with pytest.raises(ValueError, match="non-Biolink predicate"):
        from_text(
            "x affects y",
            llm,
            subject=Entity("CHEBI:1"),
            object=Entity("HGNC:1"),
        )


def test_rejects_out_of_vocabulary_qualifier():
    llm = StubLLM(
        {"predicate": "affects", "object_direction_qualifier": "sideways"}
    )
    with pytest.raises(ValueError, match="Invalid value"):
        from_text(
            "x affects y",
            llm,
            subject=Entity("CHEBI:1"),
            object=Entity("HGNC:1"),
        )


def test_categories_constrain_predicate_schema_and_prompt():
    llm = StubLLM({"predicate": "affects"})
    from_text(
        "methionine affects increased expression of ADRB2",
        llm,
        subject=Entity("CHEBI:16811", "methionine", "chemical entity"),
        object=Entity("HGNC:286", "ADRB2", "gene"),
    )
    schema = llm.last_call["schema"]
    prompt = llm.last_call["prompt"]
    # Predicate is restricted to the candidate set...
    assert "enum" in schema["properties"]["predicate"]
    assert "affects" in schema["properties"]["predicate"]["enum"]
    assert "has_phenotype" not in schema["properties"]["predicate"]["enum"]
    # ...and the prompt surfaces candidates plus qualifier descriptions.
    assert "Candidate predicates" in prompt
    assert "chemical entity" in prompt  # subject category shown
    assert "aspect" in prompt.lower()  # qualifier description present


def test_uncategorized_entities_leave_predicate_free():
    llm = StubLLM({"predicate": "affects"})
    from_text(
        "x affects y",
        llm,
        subject=Entity("CHEBI:1"),
        object=Entity("HGNC:1"),
    )
    assert "enum" not in llm.last_call["schema"]["properties"]["predicate"]
    assert "Candidate predicates" not in llm.last_call["prompt"]


def test_rejects_predicate_outside_candidate_set():
    # 'has_phenotype' is a real Biolink predicate but cannot connect chem -> gene.
    llm = StubLLM({"predicate": "has_phenotype"})
    with pytest.raises(ValueError, match="cannot connect"):
        from_text(
            "methionine affects ADRB2",
            llm,
            subject=Entity("CHEBI:16811", "methionine", "chemical entity"),
            object=Entity("HGNC:286", "ADRB2", "gene"),
        )


def test_round_trip_recovers_qualifiers():
    """to_text -> from_text recovers the qualifier set (with a stub LLM)."""
    original = QualifiedStatement(
        subject=Entity("CHEBI:33216", "Bisphenol A"),
        predicate="affects",
        qualified_predicate="biolink:causes",
        object=Entity("HGNC:3467", "ESR1"),
        object_aspect_qualifier="degradation",
        object_direction_qualifier="decreased",
    )
    # A faithful LLM would extract exactly these qualifier values.
    llm = StubLLM(
        {
            "predicate": original.predicate,
            "qualified_predicate": original.qualified_predicate,
            "object_aspect_qualifier": original.object_aspect_qualifier,
            "object_direction_qualifier": original.object_direction_qualifier,
        }
    )
    recovered = from_text(
        to_text(original),
        llm,
        subject=original.subject,
        object=original.object,
    )
    assert recovered.qualifiers() == original.qualifiers()
