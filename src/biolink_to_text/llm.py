"""Provider-agnostic LLM interface for the text -> triple direction.

This library ships no provider SDK. Callers implement :class:`LLMClient` around
whatever backend they use (Anthropic, OpenAI, a ``litellm`` wrapper, a local
model, or a test stub) and pass it to :func:`biolink_to_text.from_text`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal structured-output contract for qualifier extraction.

    Implementations must return a JSON object (as a ``dict``) conforming to
    ``response_schema``. They are responsible for transport, auth, retries, and
    coercing the model's reply into a dict.
    """

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        response_schema: dict,
    ) -> dict:
        """Return a structured object answering ``prompt`` under ``system``."""
        ...
