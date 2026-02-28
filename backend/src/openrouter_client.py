"""
OpenRouter API client
Provides a thin wrapper around the OpenRouter REST API (OpenAI-compatible).

Requires the OPENROUTER_API_KEY environment variable to be set.
"""
import json
import os
import requests
from typing import Generator


class OpenRouterClient:
    """Reusable HTTP client for OpenRouter chat completions."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: str = "http://localhost",
        site_name: str = "Scabber Voice Assistant",
    ):
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Export it with: export OPENROUTER_API_KEY=sk-or-..."
            )
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            }
        )

    # ------------------------------------------------------------------
    # Non-streaming completion
    # ------------------------------------------------------------------\
    def chat_complete(
        self,
        model: str,
        messages: list,
        temperature: float = 0.5,
        max_tokens: int = 512,
        timeout: int = 120,
    ) -> str:
        """
        Send a chat completion request and return the full response text.

        Args:
            model: OpenRouter model identifier, e.g. "qwen/qwen3-0.6b".
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.

        Returns:
            The assistant's reply text.

        Raises:
            requests.HTTPError: On non-2xx responses.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        response = self._session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Streaming completion
    # ------------------------------------------------------------------
    def chat_complete_stream(
        self,
        model: str,
        messages: list,
        temperature: float = 0.5,
        max_tokens: int = 512,
        timeout: int = 120,
    ) -> Generator[str, None, None]:
        """
        Send a streaming chat completion request.

        Yields:
            Text chunks as they arrive from the server.

        Raises:
            requests.HTTPError: On non-2xx responses.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        response = self._session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=timeout,
            stream=True,
        )
        response.raise_for_status()

        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                content = data["choices"][0].get("delta", {}).get("content")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
