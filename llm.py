"""
Phase 1 - Project 1: a multi-provider LLM abstraction.

The idea: your app should not care *which* provider it talks to. Define one
small interface, implement it once per provider, and swapping providers
becomes a one-line change. This is the pattern behind real production code -
and the "multi-provider abstraction layer" line on your resume.

You will finish OllamaProvider (see the TODO at the bottom of this file).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Iterator

from dotenv import load_dotenv

load_dotenv()


class LLMProvider(ABC):
    """Every provider implements the same interface."""

    name: str = "base"

    @abstractmethod
    def stream(self, messages: list[dict], temperature: float = 0.7) -> Iterator[str]:
        """Yield the response text in chunks as it is generated."""
        raise NotImplementedError

    def complete(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Convenience: collect the stream into a single string."""
        return "".join(self.stream(messages, temperature))


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self, model: str = "openai/gpt-oss-120b"):
        from groq import Groq

        key = os.getenv("GROQ_API_KEY")
        if not key or key == "your_key_here":
            raise RuntimeError("GROQ_API_KEY missing - check your .env file.")
        self.client = Groq(api_key=key)
        self.model = model

    def stream(self, messages: list[dict], temperature: float = 0.7) -> Iterator[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in response:
            piece = chunk.choices[0].delta.content or ""
            if piece:
                yield piece


class OllamaProvider(LLMProvider):
    """Runs a model on your own machine - free, offline, no API key.

    Install Ollama from https://ollama.com, then pull a model:
        ollama pull llama3.2
    """

    name = "ollama"

    def __init__(self, model: str = "hf.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF:latest"):
        self.model = model
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def stream(self, messages: list[dict], temperature: float = 0.7) -> Iterator[str]:
        # TODO (your job): make this stream, just like GroqProvider.
        import ollama
        for chunk in ollama.chat(model=self.model, messages=messages, stream=True):
            piece = chunk["message"]["content"]
            if piece:
                yield piece
        # Two routes - pick either:
        #
        # (a) Simple: the `ollama` package (already in requirements.txt)
        #       import ollama
        #       for chunk in ollama.chat(model=self.model, messages=messages,
        #                                stream=True):
        #           piece = chunk["message"]["content"]
        #           if piece:
        #               yield piece
        #
        # (b) Stretch: raw HTTP with httpx, so you see what the SDK hides
        #       POST {self.host}/api/chat with
        #       {"model": self.model, "messages": messages, "stream": True}
        #     Each streamed line is a JSON object; the text lives in
        #     chunk["message"]["content"].
        #
        # When you're done, both providers should behave identically through
        # the same .stream() interface.
        


PROVIDERS = {
    "groq": GroqProvider,
    "ollama": OllamaProvider,
}
