"""Intent classification service.

This module provides intent classification for user messages,
determining whether they want to transform data, chat, or perform other actions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.llm.client import OpenRouterClient, get_llm_client
from services.llm.prompts import build_intent_prompt


class IntentClassifierService:
    """Service for classifying user intent.
    
    Uses an LLM to determine what the user wants to do based on their message.
    Distinguishes between data transformation commands and conversational queries.
    
    Attributes:
        _client: LLM client for making API calls.
    """
    
    def __init__(self, client: Optional[OpenRouterClient] = None, model: Optional[str] = None):
        """Initialize the intent classifier.
        
        Args:
            client: LLM client instance (uses default if None).
            model: Optional model identifier to override the default.
        """
        if client is not None:
            self._client = client
        elif model:
            # Use a dedicated client with the selected default model
            self._client = OpenRouterClient(default_model=model)
        else:
            self._client = get_llm_client()
    
    async def classify(
        self,
        message: str,
        has_data: bool = False,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Classify the intent of a user message.
        
        Args:
            message: User's input message.
            has_data: Whether user has loaded data.
            columns: Available column names if data is loaded.
        
        Returns:
            Dictionary containing:
            - intent: The classified intent ('transform_data', 'chat', etc.)
            - params: Extracted parameters for the intent
            - explanation: Brief explanation of the classification
            - error: Optional error message if classification failed
        """
        columns = columns or []
        system_prompt = build_intent_prompt(has_data, columns)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]
        
        response = await self._client.complete(
            messages, 
            temperature=0.1, 
            max_tokens=500,
        )
        
        result = self._client.parse_json_response(response)
        
        if result and "intent" in result:
            return result
        
        return {
            "intent": "chat",
            "params": {},
            "explanation": "Failed to parse intent from LLM response",
            "error": "Invalid LLM response format",
        }


# Convenience function for backward compatibility
async def classify_intent(
    user_message: str,
    has_data: bool = False,
    columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Classify user intent (convenience function).
    
    Args:
        user_message: User's input message.
        has_data: Whether user has loaded data.
        columns: Available column names.
    
    Returns:
        Intent classification result.
    """
    service = IntentClassifierService()
    return await service.classify(user_message, has_data, columns)
