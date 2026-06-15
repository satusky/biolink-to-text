# biolink-to-text

Library for inter-converting knowledge graph (KG) triples/paths in the Biolink Model and natural language text


## Challenges

The [Biolink Model](https://biolink.github.io/biolink-model/) is a data model for biomedical KGs, where pieces of knowledge are represented as (subject, predicate, object) triples. The nodes (entities) are relatively straightforward, since they can be mapped to ontological identifiers. However, edges (predicates) often have qualifiers that capture the details and nuance of relationships between the two entities (e.g. directionality of an effect, the attribute of the entity involved, etc.). A full list of qualifiers is available at [https://biolink.github.io/biolink-model/#qualifiers-visualization].

Construction of a natural language representation of a qualified Biolink predicate is challenging because it potentially requires not only domain knowledge but also knowledge of the proper order in which to include qualifiers to achieve proper syntax and semantics. Additionally, there can be multiple valid ways to compose a statement from a given predicate. For examples, see [Associations with Qualifiers](https://biolink.github.io/biolink-model/association-examples-with-qualifiers/) and [Interpeting a Fully Qualified Edge](https://biolink.github.io/biolink-model/reading-a-qualifier-based-statement/).

These challenges are compounded when composing a Biolink triple with a qualified predicate from natural language statements. Again, identification and ontological mapping of named entities is relatively simple, but parsing whether/which attributes of the entities specifically are participating in the relationship, directionality, and any other details into proper Bioink qualifiers.


## Goals

Create a library of tools for converting between Biolink Model triples and natural language.


## Installation

Use `uv` for package management.

```bash
uv sync --extra dev
```

## Usage

### Triple → text

Build a `QualifiedStatement` and render it. Qualifiers compose in Biolink's
strict order to produce a grammatical sentence:

```python
from biolink_to_text import Entity, QualifiedStatement, to_text

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

to_text(stmt)
# "Hexachlorobenzene metabolite causes increased methylation of a mutant
#  form of the CDKN2A promoter in the nucleus of HeLa cells"
```

Entity labels are caller-supplied; without one, the CURIE is used as the
surface form. Predicate/qualifier vocabularies are sourced from the Biolink
Model Toolkit (`bmt`).

### Text → triple

`from_text` extracts the predicate and qualifiers connecting two
**already-resolved** entities (named-entity recognition and ontology mapping
are out of scope — supply the CURIEs from your own tooling). It is
provider-agnostic: pass any object implementing the `LLMClient` protocol
(`complete(*, system, prompt, response_schema) -> dict`).

```python
from biolink_to_text import Entity, from_text

stmt = from_text(
    "Bisphenol A causes decreased degradation of ESR1",
    my_llm_client,  # your LLMClient implementation
    subject=Entity("CHEBI:33216", "Bisphenol A"),
    object=Entity("HGNC:3467", "ESR1"),
)
```

The returned predicate and enum-valued qualifiers are validated against the
Biolink Model; out-of-vocabulary values raise `ValueError`.

## Development

```bash
uv run pytest      # tests (golden triple→text + stubbed text→triple)
uv run ruff check  # lint
```
