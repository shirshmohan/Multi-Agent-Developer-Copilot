"""
llm/providers.py
Concrete backends. Each one's ONLY job: call its API and translate the result
into the shared LLMResponse shape from base.py.
"""
import json
import uuid
from .base import LLMProvider, LLMResponse, ToolCall


class OpenAIProvider(LLMProvider):
    """Talks to the OpenAI API. The only file in the project that imports openai."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        from openai import OpenAI                 # imported lazily so Ollama users need not install it
        self._client = OpenAI(api_key=api_key)    # falls back to OPENAI_API_KEY env var
        self._model = model

    def generate(self, messages, tools=None):
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, tools=tools)
        msg = resp.choices[0].message

        # --- normalize: OpenAI gives a structured tool_calls array ---
        calls = []
        for tc in (msg.tool_calls or []):
            calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),  # provider hands args as a JSON string
            ))
        return LLMResponse(text=msg.content, tool_calls=calls, raw=resp)


class OllamaProvider(LLMProvider):
    """Talks to a local Ollama server (e.g. running qwen2.5). Same contract, different wire format."""

    def __init__(self, model: str = "qwen2.5:3b", host: str = "http://localhost:11434"):
        import ollama                             # lazy import, only if you actually use local
        self._client = ollama.Client(host=host)
        self._model = model

    def generate(self, messages, tools=None):
        resp = self._client.chat(model=self._model, messages=messages, tools=tools)
        msg = resp["message"]

        # --- normalize: Ollama nests tool calls differently and omits ids ---
        calls = []
        for tc in msg.get("tool_calls", []) or []:
            fn = tc["function"]
            args = fn["arguments"]
            if isinstance(args, str):             # some versions return a string, some a dict
                args = json.loads(args)
            calls.append(ToolCall(
                id=str(uuid.uuid4()),             # Ollama gives no id, so we synthesize one
                name=fn["name"],
                arguments=args,
            ))
        return LLMResponse(text=msg.get("content"), tool_calls=calls, raw=resp)
