"""Chat service for data-aware conversations.

This module provides the chat service that handles user conversations
about datasets, using function calling for data queries.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

from config import settings
from services.llm.client import OpenRouterClient, get_llm_client
from services.llm.prompts import (
    build_chat_prompt,
    build_query_classifier_prompt,
    GENERAL_CONVERSATION_TEMPLATE,
)
from services.llm.tools import CHAT_TOOLS
from services.queries import get_query_registry

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class ChatService:
    """Service for handling data-aware chat interactions.
    
    Provides conversational AI capabilities for querying and discussing
    datasets, with function calling support for data operations.
    
    Attributes:
        _client: LLM client for API calls.
        _query_registry: Registry of query handlers.
    """
    
    def __init__(self, client: Optional[OpenRouterClient] = None, model: Optional[str] = None):
        """Initialize the chat service.
        
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
        self._query_registry = get_query_registry()
    
    async def chat(
        self,
        user_message: str,
        data_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        session: Optional["AsyncSession"] = None,
        dataset_id: Optional[str] = None,
    ) -> str:
        """Chat with the dataset curator agent.
        
        Args:
            user_message: User's input message.
            data_path: Path to the data file (if any).
            context: Additional context (columns, etc.).
            history: Conversation history.
            session: Database session for semantic search.
            dataset_id: Dataset ID for semantic search.
        
        Returns:
            Assistant's response message.
        """
        df, columns, is_large = await self._prepare_context(data_path, context)
        
        query_type = await self._classify_query_type(user_message, columns)
        print(f"[Chat] Query: {user_message[:50]}... | Type: {query_type}")
        
        if query_type == "general":
            return await self._handle_general_conversation(user_message, history)
        
        if df is not None:
            return await self._handle_data_query_with_tools(
                df, user_message, columns, history, context, session, dataset_id, data_path
            )
        
        # No data available, use regular chat
        system_prompt = build_chat_prompt(columns)
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return await self._client.complete(messages, temperature=0.7)
    
    async def _prepare_context(
        self,
        data_path: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> tuple[Optional[pd.DataFrame], List[str], bool]:
        """Prepare DataFrame and columns for chat.
        
        Args:
            data_path: Path to data file.
            context: Additional context.
        
        Returns:
            Tuple of (DataFrame, columns, is_large).
        """
        from data_loader import is_large_dataset, load_dataframe_smart
        
        is_large = False
        df = None
        
        if data_path:
            is_large, _ = await is_large_dataset(data_path)
            max_rows = settings.data_loader.sample_size if is_large else None
            df = await load_dataframe_smart(data_path, max_rows=max_rows, sample=is_large)
        
        columns = context.get("columns", []) if context else []
        if df is not None and not columns:
            columns = list(df.columns)
        
        return df, columns, is_large
    
    async def _classify_query_type(
        self,
        user_message: str,
        columns: List[str],
    ) -> str:
        """Classify if query is about data or general conversation.
        
        Args:
            user_message: User's message.
            columns: Available columns.
        
        Returns:
            'data' or 'general'.
        """
        prompt = build_query_classifier_prompt(columns)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ]
        
        response = await self._client.complete(messages, temperature=0, max_tokens=10)
        result = response.strip().lower()
        return "data" if "data" in result else "general"
    
    async def _handle_general_conversation(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> str:
        """Handle general conversation.
        
        Args:
            user_message: User's message.
            history: Conversation history.
        
        Returns:
            Response message.
        """
        messages = [{"role": "system", "content": GENERAL_CONVERSATION_TEMPLATE}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return await self._client.complete(messages, temperature=0.7)
    
    async def _handle_data_query_with_tools(
        self,
        df: pd.DataFrame,
        user_message: str,
        columns: List[str],
        history: Optional[List[Dict[str, str]]],
        context: Optional[Dict[str, Any]],
        session: Optional["AsyncSession"],
        dataset_id: Optional[str],
        data_path: Optional[str],
    ) -> str:
        """Handle data queries using function calling.
        
        Args:
            df: DataFrame to query.
            user_message: User's message.
            columns: Available columns.
            history: Conversation history.
            context: Additional context.
            session: Database session.
            dataset_id: Dataset ID.
            data_path: Path to data file.
        
        Returns:
            Response message.
        """
        system_prompt = build_chat_prompt(columns)
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = await self._client.complete_with_tools(
                messages,
                tools=CHAT_TOOLS,
                temperature=0.7,
                max_tokens=2048,
            )
            
            message = response.choices[0].message
            print(f"[Chat] Initial response: content={repr(message.content)}, tool_calls={message.tool_calls}")
            
            if not message.content and not message.tool_calls:
                print("[Chat] Empty response with no tools, retrying without function calling...")
                return await self._chat_without_tools(df, user_message, context, session, dataset_id, data_path)

            # If the model returned text but no tool calls, it might not have understood
            # that it should query the data. Check if this looks like a data query that
            # should have used tools.
            if message.content and not message.tool_calls:
                content_lower = message.content.lower()
                # Detect phrases indicating the model couldn't access data
                no_data_phrases = [
                    "i don't have access",
                    "i cannot provide",
                    "data is not provided",
                    "without more information",
                    "i can't directly",
                    "i cannot directly",
                    "not provided to me",
                    "i don't have the actual",
                    "i cannot see the actual",
                ]
                if any(phrase in content_lower for phrase in no_data_phrases):
                    print("[Chat] Model claims no data access - forcing tool-based query...")
                    # Add stronger instruction and retry
                    messages.append({
                        "role": "user",
                        "content": f"You DO have access to query the data using the available functions. Please use the appropriate function (like get_distinct_values, get_statistics, group_by, etc.) to answer: {user_message}"
                    })
                    retry_response = await self._client.complete_with_tools(
                        messages,
                        tools=CHAT_TOOLS,
                        temperature=0.3,
                        max_tokens=2048,
                    )
                    message = retry_response.choices[0].message
                    print(f"[Chat] Retry response: content={repr(message.content)}, tool_calls={message.tool_calls}")

            message = await self._execute_tool_calls(df, message, messages)
            
            if message.content:
                return message.content
            
            print("[Chat] No content after tool calls, falling back...")
            return await self._chat_without_tools(df, user_message, context, session, dataset_id, data_path)
            
        except Exception as e:
            error_str = str(e).lower()
            print(f"[Chat] Error: {e}")
            
            if any(indicator in error_str for indicator in ["tool", "function", "no endpoints found"]):
                print("[Chat] Tool calling not supported, falling back to non-tool chat...")
                return await self._chat_without_tools(df, user_message, context, session, dataset_id, data_path)
            
            raise
    
    async def _execute_tool_calls(
        self,
        df: pd.DataFrame,
        message: Any,
        messages: List[Dict[str, Any]],
    ) -> Any:
        """Execute tool calls and return updated message.
        
        Args:
            df: DataFrame to query.
            message: Message with tool calls.
            messages: Conversation messages.
        
        Returns:
            Final message after tool execution.
        """
        iteration = 0
        max_iterations = settings.llm.max_tool_iterations
        
        while message.tool_calls and iteration < max_iterations:
            iteration += 1
            messages.append(message)
            
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                
                print(f"[Chat] Calling function: {function_name}({arguments})")
                result = self._query_registry.execute(df, function_name, arguments)
                print(f"[Chat] Result: {result}")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })
            
            response = await self._client.complete_with_tools(
                messages,
                tools=CHAT_TOOLS,
                temperature=0.7,
                max_tokens=2048,
            )
            message = response.choices[0].message
        
        return message

    async def _chat_without_tools(
        self,
        df: Optional[pd.DataFrame],
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        session: Optional["AsyncSession"] = None,
        dataset_id: Optional[str] = None,
        data_path: Optional[str] = None,
    ) -> str:
        """Fallback chat without function calling.
        
        Args:
            df: DataFrame to query.
            user_message: User's message.
            context: Additional context.
            session: Database session.
            dataset_id: Dataset ID.
            data_path: Path to data file.
        
        Returns:
            Response message.
        """
        columns = context.get("columns", []) if context else (list(df.columns) if df is not None else [])
        
        stats = {
            "rows": len(df) if df is not None else 0,
            "columns": len(df.columns) if df is not None else len(columns),
            "column_names": columns[:15],
        }
        
        # Try semantic search if available
        query_results = await self._try_semantic_search(session, dataset_id, user_message)
        
        if not query_results:
            query_results = await self._get_sample_data_results(df, data_path)
        
        system_prompt = self._build_data_response_prompt(stats, query_results)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        
        return await self._client.complete(messages, temperature=0.3)
    
    async def _try_semantic_search(
        self,
        session: Optional["AsyncSession"],
        dataset_id: Optional[str],
        user_message: str,
    ) -> Optional[str]:
        """Try semantic search for query results.
        
        Args:
            session: Database session.
            dataset_id: Dataset ID.
            user_message: User's message.
        
        Returns:
            Formatted search results or None.
        """
        if not session or not dataset_id:
            return None
        
        try:
            from embeddings import semantic_search, has_embeddings
            if not await has_embeddings(session, dataset_id):
                return None
            
            print(f"[Chat] Using semantic search for: {user_message[:50]}...")
            results = await semantic_search(
                session, dataset_id, user_message, 
                limit=settings.data_loader.search_result_limit
            )
            
            if not results:
                return None
            
            query_results = "SEMANTICALLY RELEVANT RESULTS:\n"
            for i, r in enumerate(results, 1):
                similarity = r["similarity"]
                metadata = r.get("metadata", {})
                title = metadata.get("Question Title", metadata.get("title", "Unknown"))
                query_results += f"{i}. **{title}** (similarity: {similarity:.2f})\n"
                query_results += f"   {r['content'][:300]}...\n\n"
            return query_results
        except Exception as e:
            print(f"[Chat] Semantic search failed: {e}")
            return None
    
    async def _get_sample_data_results(
        self,
        df: Optional[pd.DataFrame],
        data_path: Optional[str],
    ) -> str:
        """Get sample data as fallback query result.
        
        Args:
            df: DataFrame to sample from.
            data_path: Path to data file.
        
        Returns:
            Formatted sample data string.
        """
        if df is not None and len(df) > 0:
            sample_rows = df.head(3).to_dict(orient="records")
            return "SAMPLE DATA:\n" + "\n".join([str(row) for row in sample_rows])
        
        if data_path:
            from data_loader import load_dataframe_smart
            sample_df = await load_dataframe_smart(data_path, max_rows=3)
            if sample_df is not None and len(sample_df) > 0:
                sample_rows = sample_df.to_dict(orient="records")
                return "SAMPLE DATA:\n" + "\n".join([str(row) for row in sample_rows])
            return "DATASET INFO: Large dataset - queries will be processed efficiently using chunked operations."
        
        return "DATASET INFO: No data available."
    
    def _build_data_response_prompt(
        self,
        stats: Dict[str, Any],
        query_results: str,
    ) -> str:
        """Build system prompt for data responses.
        
        Args:
            stats: Dataset statistics.
            query_results: Query results to include.
        
        Returns:
            System prompt string.
        """
        return f"""You are the Dataset Curator assistant. Answer the user's question using ONLY the data provided below.

Dataset info: {stats['rows']} rows, {stats['columns']} columns
Columns: {', '.join(stats['column_names'])}

QUERY RESULTS:
{query_results}

RULES:
1. Use ONLY the data shown above
2. Be direct and concise
3. Format numbers nicely"""


# Convenience function for backward compatibility
async def chat_with_agent(
    user_message: str,
    data_path: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    session: Optional["AsyncSession"] = None,
    dataset_id: Optional[str] = None,
) -> str:
    """Chat with the dataset curator agent (convenience function).
    
    Args:
        user_message: User's input message.
        data_path: Path to data file.
        context: Additional context.
        history: Conversation history.
        session: Database session.
        dataset_id: Dataset ID.
    
    Returns:
        Assistant's response message.
    """
    service = ChatService()
    return await service.chat(
        user_message, data_path, context, history, session, dataset_id
    )
