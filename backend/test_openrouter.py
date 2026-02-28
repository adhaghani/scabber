"""Quick end-to-end test for the OpenRouter integration."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import OPENROUTER_API_KEY, OPENROUTER_MODEL, SMART_MODEL
from openrouter_client import OpenRouterClient

client = OpenRouterClient(api_key=OPENROUTER_API_KEY)

# ── 1. Non-streaming ──────────────────────────────────────────────────────────
print("=== Non-streaming test ===")
reply = client.chat_complete(
    model=OPENROUTER_MODEL,
    messages=[{"role": "user", "content": "Say exactly: Hello from OpenRouter!"}],
    max_tokens=20,
)
print("Reply:", reply)

# ── 2. Streaming ──────────────────────────────────────────────────────────────
print("\n=== Streaming test ===")
chunks = []
for chunk in client.chat_complete_stream(
    model=OPENROUTER_MODEL,
    messages=[{"role": "user", "content": "Count to 3, one number per line."}],
    max_tokens=30,
):
    print(chunk, end="", flush=True)
    chunks.append(chunk)
print(f"\nTotal chunks received: {len(chunks)}")

# ── 3. LanguageModel high-level API ──────────────────────────────────────────
print("\n=== LanguageModel.generate_response test ===")
from llm_module import LanguageModel
lm = LanguageModel()
lm.load()
response = lm.generate_response("What is 2 + 2?", use_history=False)
print("Response:", response)

print("\nAll tests passed!")
