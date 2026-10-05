"""
Phase 0 - Your first LLM call.

Goal: prove your environment works end to end - Python, a virtual env,
a secret stored in .env, and one real call to a free LLM API (Groq).

Run it with:   python first_call.py
"""

import os
import sys

from dotenv import load_dotenv
from groq import Groq

# Load variables from .env into the environment
load_dotenv()

MODEL = "openai/gpt-oss-120b"


QUESTION = (
    "In two sentences, explain what an LLM is to someone who has never used one."
)


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        sys.exit(
            "No GROQ_API_KEY found.\n"
            "1) Copy .env.example to .env\n"
            "2) Paste your key from https://console.groq.com/keys\n"
            "3) Run this script again."
        )

    client = Groq(api_key=api_key)

    print(f"Model:    {MODEL}")
    print(f"Question: {QUESTION}")
    print()
    print("Answer (streaming): ", end="", flush=True)

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": QUESTION}],
        temperature=0.7,
        stream=True,
    )

    for chunk in stream:
        piece = chunk.choices[0].delta.content or ""
        print(piece, end="", flush=True)

    print()
    print()
    print("Done. If you saw a real answer above, Phase 0 works.")


if __name__ == "__main__":
    main()
