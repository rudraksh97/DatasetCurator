"""Execution planning service.

This module provides multi-step execution planning for data transformations,
breaking down complex user requests into atomic operations.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.llm.client import OpenRouterClient, get_llm_client
from services.llm.prompts import build_planner_prompt


class ExecutionPlannerService:
    """Service for creating execution plans from user requests.
    
    Breaks down complex transformation requests into a sequence of
    atomic, executable steps.
    
    Attributes:
        _client: LLM client for making API calls.
    """
    
    def __init__(self, client: Optional[OpenRouterClient] = None):
        """Initialize the execution planner.
        
        Args:
            client: LLM client instance (uses default if None).
        """
        self._client = client or get_llm_client()
    
    async def create_plan(
        self,
        user_message: str,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create an execution plan from user request.
        
        Args:
            user_message: User's transformation request.
            columns: Available column names.
        
        Returns:
            Dictionary containing:
            - is_multi_step: Whether the plan has multiple steps
            - steps: List of step dictionaries with operation details
            - error: Optional error message if planning failed
        """
        columns = columns or []
        system_prompt = build_planner_prompt(columns)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        
        response = await self._client.complete(
            messages,
            temperature=0.1,
            max_tokens=1000,
        )
        
        result = self._client.parse_json_response(response)
        
        if result and "steps" in result:
            # Ensure is_analysis is present (default to false if not specified)
            if "is_analysis" not in result:
                result["is_analysis"] = False
            return result
        
        return {
            "is_multi_step": False,
            "is_analysis": False,
            "steps": [],
            "error": "Failed to parse execution plan from LLM response",
        }


# Convenience function for backward compatibility
async def create_execution_plan(
    user_message: str,
    columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create execution plan (convenience function).
    
    Args:
        user_message: User's transformation request.
        columns: Available column names.
    
    Returns:
        Execution plan dictionary.
    """
    service = ExecutionPlannerService()
    return await service.create_plan(user_message, columns)
