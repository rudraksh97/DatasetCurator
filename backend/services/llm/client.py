"""LLM client implementations.

This module provides LLM client implementations that conform to the LLMClient protocol.
Currently supports OpenRouter, but can be extended for other providers.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

from openai import AsyncOpenAI, RateLimitError, APIError

from config import settings
from protocols import Message


class LLMRateLimitError(Exception):
    """Raised when the LLM API rate limit is exceeded."""
    pass


class LLMAPIError(Exception):
    """Raised when the LLM API returns an error."""
    pass


# Static list of known free models on OpenRouter that work well with this app.
# These IDs must match OpenRouter's model identifiers.
FREE_LLM_MODELS: List[Dict[str, Any]] = [
    {
        "id": "meta-llama/llama-3.3-70b-instruct:free",
        "name": "Llama 3.3 70B (free)",
        "provider": "meta",
        "context_length": 128_000,
        "is_default": True,
    },
    {
        "id": "mistralai/mistral-7b-instruct:free",
        "name": "Mistral 7B Instruct (free)",
        "provider": "mistral",
        "context_length": 32_000,
        "is_default": False,
    },
    {
        "id": "deepseek/deepseek-r1-0528:free",
        "name": "DeepSeek R1 0528 (free)",
        "provider": "deepseek",
        "context_length": 163_840,
        "is_default": False,
    },
    {
        "id": "minimax/minimax-m2:free",
        "name": "MiniMax M2 (free)",
        "provider": "minimax",
        "context_length": 204_800,
        "is_default": False,
    },
]


def _parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response, handling markdown code blocks.
    
    Args:
        response: Raw response string from LLM.
    
    Returns:
        Parsed JSON dictionary or None if parsing fails.
    """
    response = response.strip()
    
    # Strip markdown code blocks if present
    if response.startswith("```"):
        parts = response.split("```")
        if len(parts) >= 2:
            response = parts[1]
            if response.startswith("json"):
                response = response[4:]
    
    response = response.strip()
    
    # Robust extraction: find matching brace
    start_idx = response.find("{")
    if start_idx != -1:
        brace_count = 0
        end_idx = -1
        
        for i, char in enumerate(response[start_idx:], start=start_idx):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                
            if brace_count == 0:
                end_idx = i
                break
        
        if end_idx != -1:
            response = response[start_idx : end_idx + 1]
        
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print(f"[LLM] JSON parse error on: {response}")
        return None


class OpenRouterClient:
    """LLM client implementation for OpenRouter API.
    
    This client wraps the OpenAI SDK configured for OpenRouter,
    providing both standard completion and tool-calling capabilities.
    
    Attributes:
        _client: The underlying AsyncOpenAI client.
        _default_model: Default model to use for completions.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        """Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (uses settings if None).
            base_url: API base URL (uses settings if None).
            default_model: Default model identifier (uses settings if None).
        
        Raises:
            ValueError: If API key is not provided or found in settings.
        """
        api_key = api_key or settings.llm.api_key
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment")
        
        base_url = base_url or settings.llm.base_url
        self._default_model = default_model or settings.llm.default_model
        
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    
    async def complete(
        self,
        messages: Sequence[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send a chat completion request.
        
        Args:
            messages: Conversation history.
            model: Model identifier (uses default if None).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
        
        Returns:
            The model's response text.
        
        Raises:
            LLMRateLimitError: If rate limit is exceeded.
            LLMAPIError: If the API returns an error.
        """
        try:
            response = await self._client.chat.completions.create(
                model=model or self._default_model,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except RateLimitError as e:
            raise LLMRateLimitError(
                "⏳ **Rate limit exceeded.** The AI service is temporarily unavailable. "
                "Please wait a moment and try again."
            ) from e
        except APIError as e:
            raise LLMAPIError(f"AI service error: {str(e)}") from e
    
    async def complete_with_tools(
        self,
        messages: Sequence[Message],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Any:
        """Send a chat completion request with function calling.
        
        Args:
            messages: Conversation history.
            tools: Tool definitions for function calling.
            model: Model identifier (uses default if None).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
        
        Returns:
            The API response object with potential tool calls.
        
        Raises:
            LLMRateLimitError: If rate limit is exceeded.
            LLMAPIError: If the API returns an error.
        """
        try:
            response = await self._client.chat.completions.create(
                model=model or self._default_model,
                messages=list(messages),
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return response
        except RateLimitError as e:
            raise LLMRateLimitError(
                "⏳ **Rate limit exceeded.** The AI service is temporarily unavailable. "
                "Please wait a moment and try again."
            ) from e
        except APIError as e:
            raise LLMAPIError(f"AI service error: {str(e)}") from e
    
    def parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse a JSON response from the LLM.
        
        Args:
            response: Raw response string.
        
        Returns:
            Parsed dictionary or None.
        """
        return _parse_json_response(response)


@lru_cache(maxsize=1)
def get_llm_client() -> OpenRouterClient:
    """Get the singleton LLM client instance.
    
    Returns:
        Configured OpenRouterClient instance.
    """
    return OpenRouterClient()
