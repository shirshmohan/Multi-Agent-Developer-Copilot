"""
llm/__init__.py
The factory. Agents call get_llm() and get a provider WITHOUT knowing which one.

Selection order:
  1. explicit argument:        get_llm("openai")
  2. env var LLM_PROVIDER:      set it per-run
  3. default:                   "openai"

This is the seam. Swapping OpenAI <-> Qwen is a flag here, nothing else in the
codebase changes. That is the entire payoff of the abstraction.
"""
import os
from .base import LLMProvider, LLMResponse, ToolCall
from .providers import OpenAIProvider, OllamaProvider


def get_llm(provider: str | None = None, **kwargs) -> LLMProvider:
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()

    if provider == "openai":
        return OpenAIProvider(**kwargs)
    if provider in ("ollama", "qwen", "local"):
        return OllamaProvider(**kwargs)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


__all__ = ["get_llm", "LLMProvider", "LLMResponse", "ToolCall"]
