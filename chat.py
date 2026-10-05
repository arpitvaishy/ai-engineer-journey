"""
Phase 1 - Project 1: streaming CLI chatbot with retry.

Run:
    python chat.py            # default provider: groq
    python chat.py ollama     # your local Ollama model

Type 'exit' to quit.

This is where you learn three production habits:
  1. Streaming - print tokens as they arrive, so the app feels fast.
  2. Retry with backoff - transient network/rate-limit errors should not
     kill the request; tenacity retries with growing waits.
  3. Abstraction - chat.py never imports Groq or Ollama directly. It only
     knows LLMProvider. That is the whole point.
"""

import sys

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm import PROVIDERS

SYSTEM_PROMPT = (
    "You are a concise, friendly assistant. Answer in at most three sentences."
)


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def open_stream(provider, messages, temperature=0.7):
    """Open a stream and pull the first chunk.

    Retrying *here* covers the call that actually fails on bad networks or
    rate limits. We return the first chunk plus the live generator, so the
    caller can keep printing the rest without re-requesting.
    """
    gen = provider.stream(messages, temperature)
    first = next(gen)  # forces the request now, so tenacity can retry it
    return first, gen


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "groq"
    cls = PROVIDERS.get(which)
    if cls is None:
        sys.exit(f"Unknown provider '{which}'. Choose: {', '.join(PROVIDERS)}")

    provider = cls()
    print(f"Provider: {provider.name}  |  type 'exit' to quit\n")

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue

        history.append({"role": "user", "content": user})
        print(f"{provider.name} > ", end="", flush=True)

        try:
            first, gen = open_stream(provider, history)
        except Exception as exc:  # noqa: BLE001 - we want to survive anything here
            print(f"\n[error] {type(exc).__name__}: {exc}")
            history.pop()
            continue

        answer = first
        print(first, end="", flush=True)
        try:
            for piece in gen:
                answer += piece
                print(piece, end="", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[stream interrupted] {type(exc).__name__}: {exc}")

        print()
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
