"""OpenAI-backed :class:`~biolink_to_text.llm.LLMClient` for text -> triple.

Implements the :class:`LLMClient` contract with the OpenAI Python SDK, using
structured outputs so the reply conforms to the JSON schema ``from_text``
supplies. The SDK is an optional dependency, imported lazily, and installed via
the ``openai`` extra (``pip install biolink-to-text[openai]``).
"""

from __future__ import annotations

import json
from typing import Any


def _strict_schema(schema: dict) -> dict:
    """Adapt a response schema to OpenAI strict structured-output rules.

    Strict mode requires every property to appear in ``required``. The optional
    qualifier slots already permit ``null`` in their type, so marking them all
    required simply lets the model emit ``null`` for those that do not apply;
    ``from_text`` skips ``None`` values when building the statement.
    """
    properties = schema.get("properties", {})
    return {**schema, "required": list(properties)}


class OpenAIClient:
    """An :class:`LLMClient` backed by OpenAI Chat Completions.

    ``model`` selects the OpenAI model; extra keyword arguments are forwarded to
    every ``chat.completions.create`` call (e.g. ``temperature``). A pre-built
    SDK client may be injected as ``client`` (useful for tests or custom auth);
    otherwise a default ``openai.OpenAI()`` is created on first use, reading
    credentials from the environment.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        client: Any | None = None,
        **create_kwargs: Any,
    ) -> None:
        if client is None:
            from openai import OpenAI  # optional dependency, imported lazily

            client = OpenAI()
        self._client = client
        self._model = model
        self._create_kwargs = create_kwargs

    def complete(self, *, system: str, prompt: str, response_schema: dict) -> dict:
        """Return a structured object answering ``prompt`` under ``system``."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "qualified_statement",
                    "schema": _strict_schema(response_schema),
                    "strict": True,
                },
            },
            **self._create_kwargs,
        )
        return json.loads(response.choices[0].message.content)
