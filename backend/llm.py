"""OpenRouter LLM client for chat and data operations."""
from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"


def get_client() -> AsyncOpenAI:
    """Get async OpenAI client configured for OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set in environment")
    return AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)


async def chat_completion(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """Send a chat completion request to OpenRouter."""
    client = get_client()
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def _parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response, handling markdown code blocks."""
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


QUICK_DATASETS = ["iris", "titanic", "tips", "mpg", "penguins", "diamonds", "flights", "gapminder"]

DATA_OPERATIONS = [
    "drop_column", "rename_column", "drop_nulls", "fill_nulls", "drop_duplicates",
    "filter_rows", "drop_rows", "add_column", "add_conditional_column",
    "convert_type", "sort", "replace_values", "lowercase", "uppercase", "strip_whitespace"
]


async def classify_intent(
    user_message: str,
    has_data: bool = False,
    columns: Optional[List[str]] = None,
    has_search_results: bool = False,
) -> Dict[str, Any]:
    """Use LLM to classify user intent and extract parameters."""
    
    # Build columns info separately to avoid backslash in f-string
    columns_info = ""
    if columns:
        columns_info = "\nAvailable columns: " + ", ".join(columns)
    
    system_prompt = f"""You are an intent classifier for a dataset curator application.

IMPORTANT: Distinguish between QUESTIONS and COMMANDS!
- QUESTIONS about data should be "chat" (user wants information, not changes)
- COMMANDS to modify data should be "transform_data" (user wants to change the data)

SELECTION RULES (only if pending search results = {has_search_results}):
- If the user replies with a number (1-8) or phrases like "load 1", "use option three", "pick #4", classify as "select_result".
- Always set params to {{"selection": <integer between 1 and 8>}}. Convert spelled-out numbers to digits.
- If there are NO search results, do NOT guess "select_result"—treat it as "chat" or ask the user to search.

Classify the user's message into ONE of these intents:

1. "search_datasets" - User wants to FIND/SEARCH for new datasets
2. "select_result" - User selecting from search results (e.g., "1", "use 2", "load 3")
3. "fetch_dataset" - User wants a specific quick-access dataset (fetch titanic, get iris)
4. "list_datasets" - User wants to see available datasets
5. "show_data" - User wants to preview current data
6. "transform_data" - User gives a COMMAND to MODIFY data (remove, delete, drop, add, rename, filter)
7. "chat" - QUESTIONS about data OR general conversation (are there, how many, what, check, tell me, do I have)

CRITICAL DISTINCTION:
- "are there any nulls?" → chat (asking a QUESTION)
- "remove the nulls" → transform_data (giving a COMMAND)
- "how many rows?" → chat (QUESTION)
- "delete rows where X > 10" → transform_data (COMMAND)
- "what columns do I have?" → chat (QUESTION)
- "drop column X" → transform_data (COMMAND)
- "check for duplicates" → chat (QUESTION - they want to know, not delete)
- "remove duplicates" → transform_data (COMMAND)

Quick-access datasets: {QUICK_DATASETS}
User has data: {has_data}
Has search results: {has_search_results}{columns_info}

Respond with ONLY valid JSON:
{{"intent": "intent_name", "params": {{}}, "explanation": "Brief explanation"}}

"select_result" params MUST look like: {{"selection": 4}} (integer)
"search_datasets": {{"query": "term"}}
"fetch_dataset": {{"dataset_name": "name"}}
"transform_data": {{"operation": "op_name", ...}}

Examples:
- "1" -> {{"intent": "select_result", "params": {{"selection": 1}}, "explanation": "User selected option 1"}}
- "load 2" -> {{"intent": "select_result", "params": {{"selection": 2}}, "explanation": "User wants result #2"}}
- "option five please" -> {{"intent": "select_result", "params": {{"selection": 5}}, "explanation": "User picked result #5"}}
- "use the second one" -> {{"intent": "select_result", "params": {{"selection": 2}}, "explanation": "User selected option 2"}}
- "find datasets about weather" -> {{"intent": "search_datasets", "params": {{"query": "weather"}}, "explanation": "Search for weather datasets"}}
- "fetch titanic" -> {{"intent": "fetch_dataset", "params": {{"dataset_name": "titanic"}}, "explanation": "Get titanic dataset"}}
- "show data" -> {{"intent": "show_data", "params": {{}}, "explanation": "Preview data"}}
- "remove age column" -> {{"intent": "transform_data", "params": {{"operation": "drop_column", "column": "age"}}, "explanation": "Remove age"}}

For transform_data, params need: {{"operation": "op_name", ...}}
Operations: {DATA_OPERATIONS}

CRITICAL: Understanding "remove" vs "filter":
- "remove rows where X == Y" → {{"operation": "drop_rows", "column": "X", "operator": "==", "value": "Y"}}
- "remove X data" → {{"operation": "drop_rows", "column": "<find_column_containing_X>", "operator": "==", "value": "X"}}
- "keep only rows where X == Y" → {{"operation": "filter_rows", "column": "X", "operator": "==", "value": "Y"}}
- "show only X" → {{"operation": "filter_rows", "column": "<find_column_containing_X>", "operator": "==", "value": "X"}}

IMPORTANT: When user says "remove X data" or "remove X", you must:
1. Look at the available columns to find which column likely contains "X"
2. Use "drop_rows" operation (NOT "filter_rows")
3. Set operator to "==" and value to "X"

Examples for transform_data:
- "remove age column" → {{"operation": "drop_column", "column": "age"}}
- "remove rows where state == California" → {{"operation": "drop_rows", "column": "state", "operator": "==", "value": "California"}}
- "remove Uttar Pradesh data" → {{"operation": "drop_rows", "column": "State / Union Territory", "operator": "==", "value": "Uttar Pradesh"}}
- "remove California" → {{"operation": "drop_rows", "column": "state", "operator": "==", "value": "California"}}
- "keep only rows where age > 18" → {{"operation": "filter_rows", "column": "age", "operator": ">", "value": 18}}
- "drop nulls" → {{"operation": "drop_nulls"}}

ONLY respond with JSON."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    
    try:
        response = await chat_completion(messages, temperature=0.1, max_tokens=500)
        result = _parse_json_response(response)
        if result and "intent" in result:
            return result
    except Exception as e:
        error_str = str(e)
        print(f"Intent classification error: {e}")
        
        # If rate limited, use simple keyword fallback (no regex)
        if "429" in error_str or "rate limit" in error_str.lower():
            # Simple string-based fallback when LLM is unavailable
            msg_lower = user_message.lower()
            
            # Check for search patterns (simple string contains, no regex)
            search_keywords = ["find", "search", "get", "look for", "dataset"]
            if any(keyword in msg_lower for keyword in search_keywords):
                # Extract query: remove common prefixes
                query = user_message
                for prefix in ["find", "search for", "search", "get", "look for"]:
                    if msg_lower.startswith(prefix):
                        query = user_message[len(prefix):].strip()
                        break
                return {
                    "intent": "search_datasets",
                    "params": {"query": query or user_message},
                    "explanation": "Fallback: detected search intent"
                }
            
            # Check for selection when search results exist
            if has_search_results:
                # Check if message is just a number
                stripped = user_message.strip().rstrip(".!?")
                if stripped.isdigit():
                    return {
                        "intent": "select_result",
                        "params": {"selection": int(stripped)},
                        "explanation": "Fallback: detected numeric selection"
                    }
    
    return {"intent": "chat", "params": {}, "explanation": "Could not classify intent"}


async def chat_with_agent(
    user_message: str,
    context: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Chat with the dataset curator agent - answers questions about data."""
    system_prompt = """You are the Dataset Curator, an AI that helps with datasets.

When users ask QUESTIONS about their data, analyze and answer:
- "are there any nulls?" → Check and report null counts per column
- "how many rows?" → Report row count
- "what columns?" → List the columns
- "any duplicates?" → Check and report duplicate counts
- "describe the data" → Give summary statistics

Use the context provided to answer accurately. Be helpful and informative.
Use markdown for formatting. If you need more info, ask."""

    if context:
        cols = context.get('columns', [])
        issues = context.get('issues', [])
        system_prompt += f"""

DATA CONTEXT:
- Columns: {cols}
- Known issues: {issues}
- Use this to answer questions about the data."""

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    
    return await chat_completion(messages, temperature=0.7)
