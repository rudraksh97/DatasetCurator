"""LLM client implementations.

This module provides LLM client implementations that conform to the LLMClient protocol.
Currently supports OpenRouter, but can be extended for other providers.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

from openai import AsyncOpenAI

from config import settings
from protocols import Message


def _parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response, handling markdown code blocks.
    
    Args:
        response: Raw response string from LLM.
    
    Returns:
        Parsed JSON dictionary or None if parsing fails.
    """
    response = response.strip()
    if response.startswith("```"):
        parts = response.split("```")
        if len(parts) >= 2:
            response = parts[1]
            if response.startswith("json"):
                response = response[4:]
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
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
        """
        response = await self._client.chat.completions.create(
            model=model or self._default_model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    
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
        """
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
