"""
test_llm.py  --  prove the abstraction works before building on it.

Run with OpenAI:   python test_llm.py openai
Run with local:    python test_llm.py ollama
(or set LLM_PROVIDER and run:  python test_llm.py)

Notice: the body below NEVER mentions openai or ollama. It only uses get_llm()
and LLMResponse. That is the whole point.
"""
import sys
from llm import get_llm

def main():
    choice = sys.argv[1] if len(sys.argv) > 1 else None
    llm = get_llm(choice)                          # the ONLY line that cares about provider

    # 1. plain text generation
    resp = llm.generate([
        {"role": "user", "content": "In one sentence, what is a primary key?"}
    ])
    print("TEXT:", resp.text)

    # 2. tool calling — same interface, normalized result
    tools = [{
        "type": "function",
        "function": {
            "name": "get_row_count",
            "description": "Return the number of rows in a given table.",
            "parameters": {
                "type": "object",
                "properties": {"table": {"type": "string"}},
                "required": ["table"],
            },
        },
    }]
    resp = llm.generate(
        [{"role": "user", "content": "How many rows are in the customers table?"}],
        tools=tools,
    )
    if resp.wants_tool:
        tc = resp.tool_calls[0]
        print(f"TOOL CALL: {tc.name}({tc.arguments})")   # identical shape for BOTH providers
    else:
        print("NO TOOL CALL, text was:", resp.text)

if __name__ == "__main__":
    main()
