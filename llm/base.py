"""
llm/base.py
The CONTRACT every provider must satisfy, plus the normalized response shape.

The single most important idea in this whole layer:
no matter which provider runs, an agent receives the SAME LLMResponse object.
Provider-specific weirdness is absorbed HERE, never leaked upward.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A normalized tool-call request. Both OpenAI and Ollama get mapped into this."""
    id: str                      # unique id for this call (we synthesize one if a provider omits it)
    name: str                    # the function the model wants to run
    arguments: dict              # already-parsed args (we json.loads provider strings for you)


@dataclass
class LLMResponse:
    """The ONE shape agents consume. text XOR tool_calls will be populated."""
    text: str | None = None              # plain assistant text, if the model just talked
    tool_calls: list[ToolCall] = field(default_factory=list)  # tool requests, if it chose to act
    raw: Any = None                      # the untouched provider response, for debugging only

    @property
    def wants_tool(self) -> bool:        # convenience the agent loop reads each turn
        return len(self.tool_calls) > 0


class LLMProvider(ABC):
    """The interface. Every backend (OpenAI, Ollama/Qwen, future ones) implements this."""

    @abstractmethod
    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        """Take chat messages (+ optional tool schemas) and return a normalized LLMResponse.
        Implementations MUST translate their provider's output into LLMResponse."""
        ...
