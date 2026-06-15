"""Tests for the OpenAI-backed LLMClient, using a fake injected SDK client.

No ``openai`` package or network access is required: a stand-in client mimics
the ``chat.completions.create -> choices[0].message.content`` shape.
"""

import json
from types import SimpleNamespace

from biolink_to_text import Entity, LLMClient, OpenAIClient, from_text


class _FakeCompletions:
    def __init__(self, content: str):
        self._content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeOpenAI:
    """Minimal stand-in for ``openai.OpenAI()``."""

    def __init__(self, content: str):
        self.chat = SimpleNamespace(completions=_FakeCompletions(content))


_SCHEMA = {
    "type": "object",
    "properties": {
        "predicate": {"type": "string"},
        "object_aspect_qualifier": {"type": ["string", "null"]},
    },
    "required": ["predicate"],
    "additionalProperties": False,
}


def test_satisfies_llmclient_protocol():
    client = OpenAIClient(client=_FakeOpenAI("{}"))
    assert isinstance(client, LLMClient)


def test_complete_parses_json_and_sends_strict_schema():
    fake = _FakeOpenAI('{"predicate": "affects", "object_aspect_qualifier": null}')
    client = OpenAIClient(model="gpt-4o-mini", client=fake, temperature=0)

    result = client.complete(system="sys", prompt="hello", response_schema=_SCHEMA)
    assert result == {"predicate": "affects", "object_aspect_qualifier": None}

    sent = fake.chat.completions.calls[0]
    assert sent["model"] == "gpt-4o-mini"
    assert sent["temperature"] == 0  # forwarded create kwargs
    assert sent["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    schema_block = sent["response_format"]["json_schema"]
    assert schema_block["strict"] is True
    # Strict mode requires every property to be listed as required.
    assert set(schema_block["schema"]["required"]) == {
        "predicate",
        "object_aspect_qualifier",
    }


def test_integrates_with_from_text():
    fake = _FakeOpenAI(
        json.dumps(
            {
                "predicate": "affects",
                "qualified_predicate": "biolink:causes",
                "object_aspect_qualifier": "degradation",
                "object_direction_qualifier": "decreased",
            }
        )
    )
    client = OpenAIClient(client=fake)
    stmt = from_text(
        "Bisphenol A causes decreased degradation of ESR1",
        client,
        subject=Entity("CHEBI:33216", "Bisphenol A"),
        object=Entity("HGNC:3467", "ESR1"),
    )
    assert stmt.predicate == "affects"
    assert stmt.object_direction_qualifier == "decreased"
