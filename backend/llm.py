"""LLM helpers for intent classification and data-aware chat.

This module serves as a facade providing backward-compatible imports
from the refactored services package. New code should import directly
from the services package.

Deprecated: Import from services.llm instead.
"""
from __future__ import annotations

# Re-export all public APIs for backward compatibility
from services.llm.client import OpenRouterClient, get_llm_client
from services.llm.intent import IntentClassifierService, classify_intent
from services.llm.planner import ExecutionPlannerService, create_execution_plan
from services.llm.chat import ChatService, chat_with_agent
from services.llm.prompts import DATA_OPERATIONS
from services.llm.tools import CHAT_TOOLS

# Re-export the convenience functions that existing code uses
__all__ = [
    # Classes
    "OpenRouterClient",
    "IntentClassifierService",
    "ExecutionPlannerService",
    "ChatService",
    # Convenience functions (backward compatible)
    "classify_intent",
    "create_execution_plan",
    "chat_with_agent",
    "get_llm_client",
    # Constants
    "DATA_OPERATIONS",
    "CHAT_TOOLS",
]


# For absolute backward compatibility, expose chat_completion directly
async def chat_completion(messages, model=None, temperature=0.7, max_tokens=2048):
    """Send a chat completion request to OpenRouter.
    
    Deprecated: Use services.llm.client.OpenRouterClient.complete() instead.
    """
    client = get_llm_client()
    return await client.complete(messages, model, temperature, max_tokens)
