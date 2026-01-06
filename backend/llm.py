"""LLM helpers for intent classification and data-aware chat."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MODEL = os.getenv("DEFAULT_LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

# Query/sampling limits (configurable via environment)
LARGE_DATASET_SAMPLE_SIZE = int(os.getenv("LARGE_DATASET_SAMPLE_SIZE", "10000"))
SEARCH_RESULT_LIMIT = int(os.getenv("SEARCH_RESULT_LIMIT", "5"))
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "10"))

# ---- Types -----------------------------------------------------------------
Message = Dict[str, Any]


def get_client() -> AsyncOpenAI:
    """Return an async OpenAI client configured for OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set in environment")
    return AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)


async def chat_completion(
    messages: Sequence[Message],
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


DATA_OPERATIONS = [
    "drop_column",
    "rename_column",
    "drop_nulls",
    "fill_nulls",
    "drop_duplicates",
    "filter_rows",
    "drop_rows",
    "add_column",
    "add_conditional_column",
    "convert_type",
    "sort",
    "replace_values",
    "lowercase",
    "uppercase",
    "strip_whitespace",
]

# ---- Multi-Step Planning ---------------------------------------------------
PLANNER_SYSTEM_TEMPLATE = """You are a dataset transformation planner. Your job is to break down user requests into a sequence of atomic operations.

Given a user's request and the available columns, create a plan with individual steps.

IMPORTANT RULES:
1. Each step should be ONE atomic operation
2. Steps execute in order - later steps see results of earlier steps
3. If a request has only ONE operation, return a single step
4. Order matters: do filtering/dropping rows BEFORE column operations when possible
5. Be specific about column names and values

Available operations: {DATA_OPERATIONS}

Operation parameter formats:
- drop_column: {{"column": "col_name"}}
- rename_column: {{"old_name": "old", "new_name": "new"}}
- drop_nulls: {{}} or {{"column": "col_name"}}
- fill_nulls: {{"column": "col_name", "value": "fill_value"}}
- drop_duplicates: {{}} or {{"column": "col_name"}}
- filter_rows: {{"column": "col_name", "operator": "op", "value": "val"}} - KEEPS matching rows
- drop_rows: {{"column": "col_name", "operator": "op", "value": "val"}} - REMOVES matching rows
- add_column: {{"name": "col_name", "value": "static_value"}}
- add_conditional_column (simple): {{"name": "col_name", "condition_column": "col", "operator": "op", "threshold": val, "true_value": "yes", "false_value": "no"}}
- add_conditional_column (MULTIPLE CONDITIONS - use this for ranges like "X as A, Y as B, Z as C"):
  {{"name": "col_name", "condition_column": "col", "conditions": [
    {{"operator": "<", "value": 30, "result": "Hard"}},
    {{"operator": "between", "value1": 30, "value2": 60, "result": "Medium"}},
    {{"operator": ">", "value": 60, "result": "Easy"}}
  ]}}
- convert_type: {{"column": "col_name", "dtype": "int|float|str|datetime"}}
- sort: {{"column": "col_name", "ascending": true|false}}
- replace_values: {{"column": "col_name", "old_value": "old", "new_value": "new"}}
- lowercase: {{"column": "col_name"}}
- uppercase: {{"column": "col_name"}}
- strip_whitespace: {{"column": "col_name"}}

CRITICAL: When user wants a column with MULTIPLE categories/ranges (e.g., "< 30 as Hard, 30-60 as Medium, > 60 as Easy"):
- Use ONE add_conditional_column with "conditions" array - DO NOT create multiple steps!
- The "conditions" array handles all ranges in a single operation

Operators for filter_rows/drop_rows: ==, !=, >, <, >=, <=, contains, startswith, endswith

Available columns: {columns}

Respond with ONLY valid JSON in this format:
{{
  "is_multi_step": true/false,
  "steps": [
    {{"step": 1, "description": "Human readable description", "operation": "op_name", "params": {{...}}}}
  ]
}}

Examples:

User: "remove nulls and drop the age column"
{{
  "is_multi_step": true,
  "steps": [
    {{"step": 1, "description": "Remove rows with null values", "operation": "drop_nulls", "params": {{}}}},
    {{"step": 2, "description": "Drop the age column", "operation": "drop_column", "params": {{"column": "age"}}}}
  ]
}}

User: "drop column X"
{{
  "is_multi_step": false,
  "steps": [
    {{"step": 1, "description": "Drop column X", "operation": "drop_column", "params": {{"column": "X"}}}}
  ]
}}

User: "clean the data: remove duplicates, drop nulls, and sort by date descending"
{{
  "is_multi_step": true,
  "steps": [
    {{"step": 1, "description": "Remove duplicate rows", "operation": "drop_duplicates", "params": {{}}}},
    {{"step": 2, "description": "Remove rows with null values", "operation": "drop_nulls", "params": {{}}}},
    {{"step": 3, "description": "Sort by date in descending order", "operation": "sort", "params": {{"column": "date", "ascending": false}}}}
  ]
}}

User: "remove rows where status is inactive, then rename status to active_status"
{{
  "is_multi_step": true,
  "steps": [
    {{"step": 1, "description": "Remove rows where status is inactive", "operation": "drop_rows", "params": {{"column": "status", "operator": "==", "value": "inactive"}}}},
    {{"step": 2, "description": "Rename status column to active_status", "operation": "rename_column", "params": {{"old_name": "status", "new_name": "active_status"}}}}
  ]
}}

User: "remove nulls and add a difficulty column where score < 30 is Hard, 30 to 60 is Medium, and above 60 is Easy"
{{
  "is_multi_step": true,
  "steps": [
    {{"step": 1, "description": "Remove rows with null values", "operation": "drop_nulls", "params": {{}}}},
    {{"step": 2, "description": "Add difficulty column based on score ranges", "operation": "add_conditional_column", "params": {{
      "name": "Difficulty",
      "condition_column": "score",
      "conditions": [
        {{"operator": "<", "value": 30, "result": "Hard"}},
        {{"operator": "between", "value1": 30, "value2": 60, "result": "Medium"}},
        {{"operator": ">", "value": 60, "result": "Easy"}}
      ]
    }}}}
  ]
}}

User: "create a column called Category where price < 50 is Budget, 50-200 is Standard, over 200 is Premium"
{{
  "is_multi_step": false,
  "steps": [
    {{"step": 1, "description": "Add Category column based on price ranges", "operation": "add_conditional_column", "params": {{
      "name": "Category",
      "condition_column": "price",
      "conditions": [
        {{"operator": "<", "value": 50, "result": "Budget"}},
        {{"operator": "between", "value1": 50, "value2": 200, "result": "Standard"}},
        {{"operator": ">", "value": 200, "result": "Premium"}}
      ]
    }}}}
  ]
}}

ONLY respond with JSON."""


async def create_execution_plan(
    user_message: str,
    columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Break down a user request into executable steps."""
    
    columns_str = ", ".join(columns) if columns else "No columns available"
    
    system_prompt = PLANNER_SYSTEM_TEMPLATE.format(
        DATA_OPERATIONS=", ".join(DATA_OPERATIONS),
        columns=columns_str,
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    
    response = await chat_completion(messages, temperature=0.1, max_tokens=1000)
    result = _parse_json_response(response)
    if result and "steps" in result:
        return result
    return {
        "is_multi_step": False,
        "steps": [],
        "error": "Failed to parse execution plan from LLM response",
    }

# ---- Prompts ---------------------------------------------------------------
INTENT_SYSTEM_TEMPLATE = """You are an intent classifier for a dataset curator application.

IMPORTANT: Distinguish between QUESTIONS and COMMANDS!
- QUESTIONS about data should be "chat" (user wants information, not changes)
- COMMANDS to modify data should be "transform_data" (user wants to change the data)

Classify the user's message into ONE of these intents:

1. "transform_data" - User gives a COMMAND to MODIFY data (single OR multiple operations)
   - Examples: "remove column X", "drop nulls and sort by date", "clean the data"
2. "chat" - Everything else: QUESTIONS about data, show data requests, general conversation

CRITICAL DISTINCTION:
- "are there any nulls?" → chat (asking a QUESTION)
- "remove the nulls" → transform_data (giving a COMMAND)
- "how many rows?" → chat (QUESTION)
- "delete rows where X > 10" → transform_data (COMMAND)
- "what columns do I have?" → chat (QUESTION)
- "drop column X" → transform_data (COMMAND)
- "check for duplicates" → chat (QUESTION - they want to know, not delete)
- "remove duplicates" → transform_data (COMMAND)

User has data: {has_data}{columns_info}

Respond with ONLY valid JSON:
{{"intent": "intent_name", "params": {{}}, "explanation": "Brief explanation"}}

Examples:
- "how many rows?" -> {{"intent": "chat", "params": {{}}, "explanation": "Question about data"}}
- "remove age column" -> {{"intent": "transform_data", "params": {{"operation": "drop_column", "column": "age"}}, "explanation": "Remove age"}}

For transform_data, params should include: {{"operation": "op_name", ...}}
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
- "add column difficulty where age < 18 is child" → {{"operation": "add_conditional_column", "name": "difficulty", "condition_column": "age", "operator": "<", "threshold": 18, "true_value": "child", "false_value": "adult"}}
- "add column level where score < 30 is Hard and 30 to 60 is Medium and > 60 is Easy" → {{"operation": "add_conditional_column", "name": "level", "condition_column": "score", "conditions": [{{"operator": "<", "value": 30, "result": "Hard"}}, {{"operator": "between", "value1": 30, "value2": 60, "result": "Medium"}}, {{"operator": ">", "value": 60, "result": "Easy"}}]}}

IMPORTANT for add_conditional_column with multiple ranges:
- When user says "X to Y" or "between X and Y", use {{"operator": "between", "value1": X, "value2": Y, "result": "Value"}}
- When user says "above X" or "> X", use {{"operator": ">", "value": X, "result": "Value"}}
- When user says "below X" or "< X", use {{"operator": "<", "value": X, "result": "Value"}}
- Always include ALL conditions in the "conditions" array, ordered from lowest to highest threshold
- The column name should be descriptive (e.g., "Difficulty Level", "Category", "Status")

ONLY respond with JSON."""

CHAT_SYSTEM_TEMPLATE = """You are the Dataset Curator, an AI that helps answer questions about datasets.

IMPORTANT: When users ask questions about the data, you MUST use the provided functions to query the actual dataset.
DO NOT rely on your training data or make up values. Always call the appropriate function to get real data.

CRITICAL MULTI-STEP WORKFLOW for complex queries:
1. FIRST: Identify columns mentioned in the query using find_columns or list_columns
2. SECOND: If the query has conditions (e.g., "people doing bsc", "where X > Y"), use search_rows or get_statistics with filters to find matching rows
3. THIRD: Perform the calculation (mean, sum, count, etc.) on the filtered data

Example: "what is avg study hours of people doing bsc"
- Step 1: find_columns(keywords=["study hours", "bsc", "degree"]) -> finds "study_hours" and "degree" columns
- Step 2: search_rows(column="degree", keyword="bsc") -> finds exact values like "BSc", "B.Sc", etc.
- Step 3: get_statistics(column="study_hours", filter_column="degree", filter_value="BSc") -> calculates mean on filtered data

Available functions:
- find_columns: Find columns that match keywords (USE THIS FIRST to identify relevant columns from the query)
- search_rows: SEARCH for rows where a column CONTAINS a keyword (partial/fuzzy match) - USE THIS to find exact values for filtering!
- get_row: Get a row matching a condition (column == value) - requires EXACT match
- get_value: Get a specific value from a row (filter by one column, get another column's value) - requires EXACT match
- get_random_value: Get a random value from a column, or a random row
- calculate_ratio: Calculate ratio between two columns (optionally filtered)
- get_statistics: Get statistics for a column (count, nulls, mean, variance, std, min, max) - SUPPORTS FILTERING!
- group_by: Group data by a column and count/aggregate
- list_columns: List all columns in the dataset
- get_row_count: Get total row and column count

CRITICAL WORKFLOW for lookups by name:
1. When user mentions something by an INFORMAL name (e.g., "3 sum problem", "two sum", "bsc"), FIRST use search_rows to find the exact match
2. Look at the search results to find the EXACT value in the dataset
3. THEN use get_value, get_row, or get_statistics with that EXACT value

DO NOT skip the search step! User names are often informal and won't match exactly.

Be accurate and cite the actual data values you retrieve."""


def _build_chat_system_prompt(columns: Optional[List[str]] = None) -> str:
    """Build the system prompt for data-grounded chat."""
    prompt = CHAT_SYSTEM_TEMPLATE
    if columns:
        prompt += f"\n\nAvailable columns: {', '.join(columns)}"
    return prompt


# Tool schema for OpenAI function calling
CHAT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "find_columns",
            "description": "Find columns in the dataset that match given keywords. USE THIS FIRST when the user mentions column names in their query (e.g., 'study hours', 'degree', 'bsc'). This helps identify which columns to use for filtering and calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "array", "items": {"type": "string"}, "description": "List of keywords to search for in column names (e.g., ['study hours', 'degree', 'bsc'])"},
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_rows",
            "description": "SEARCH for rows where a column CONTAINS a keyword (case-insensitive partial match). USE THIS FIRST when user gives an informal/partial name to find the exact value before using get_row or get_value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name to search in"},
                    "keyword": {"type": "string", "description": "Keyword to search for (partial match, case-insensitive)"},
                    "limit": {"type": "integer", "description": "Max results to return (default 5)"},
                },
                "required": ["column", "keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_row",
            "description": "Get a row matching an EXACT condition. Use search_rows first if you need to find the exact value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name to filter by"},
                    "value": {"type": "string", "description": "EXACT value to match"},
                },
                "required": ["column", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_value",
            "description": "Get a specific value from a row using EXACT match. Use search_rows first to find the exact filter_value if user gave an informal name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name to get the value from"},
                    "filter_column": {"type": "string", "description": "Column name to filter by"},
                    "filter_value": {"type": "string", "description": "EXACT value to filter by"},
                },
                "required": ["column", "filter_column", "filter_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_random_value",
            "description": "Get a random value from a column, or a random row. Use for questions like 'give me a random X', 'show me a random Y', 'random question text'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Optional: Column name to get a random value from. If not provided, returns a random row."}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_ratio",
            "description": "Calculate ratio between two columns. Optionally filter by a condition first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "numerator_column": {"type": "string", "description": "Column for numerator"},
                    "denominator_column": {"type": "string", "description": "Column for denominator"},
                    "filter_column": {"type": "string", "description": "Optional: Column to filter by"},
                    "filter_value": {"type": "string", "description": "Optional: Value to filter by"},
                },
                "required": ["numerator_column", "denominator_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_statistics",
            "description": "Get statistics for a column (count, nulls, mean, variance, std, min, max). SUPPORTS FILTERING: Use filter_column and filter_value to calculate statistics on a subset of data (e.g., average study hours for people doing BSc).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name to calculate statistics for"},
                    "filter_column": {"type": "string", "description": "Optional: Column to filter by (e.g., 'degree' or 'program')"},
                    "filter_value": {"type": "string", "description": "Optional: Value to filter by (e.g., 'BSc' or 'bsc'). Use search_rows first to find the exact value if needed."},
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "group_by",
            "description": "Group data by a column and count or aggregate. Use this for questions like 'breakdown by X', 'group by Y', 'count by Z', 'how many of each X'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name to group by"},
                    "aggregation": {"type": "string", "description": "Aggregation function: 'count' (default), 'sum', 'mean', 'min', 'max'"},
                    "aggregation_column": {"type": "string", "description": "Optional: Column to aggregate when using sum/mean/min/max"},
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_columns",
            "description": "List all columns in the dataset",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_row_count",
            "description": "Get total number of rows and columns",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


async def classify_intent(
    user_message: str,
    has_data: bool = False,
    columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Use LLM to classify user intent and extract parameters."""
    
    columns_info = ""
    if columns:
        columns_info = "\nAvailable columns: " + ", ".join(columns)

    system_prompt = INTENT_SYSTEM_TEMPLATE.format(
        has_data=has_data,
        columns_info=columns_info,
        DATA_OPERATIONS=", ".join(DATA_OPERATIONS),
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    
    response = await chat_completion(messages, temperature=0.1, max_tokens=500)
    result = _parse_json_response(response)
    if result and "intent" in result:
        return result
    return {
        "intent": "chat",
        "params": {},
        "explanation": "Failed to parse intent from LLM response",
        "error": "Invalid LLM response format",
    }


def _load_dataframe(data_path: Optional[str], max_rows: Optional[int] = None, sample: bool = False) -> Optional[pd.DataFrame]:
    """Load DataFrame from path with smart handling for large files."""
    if not data_path:
        return None
    from data_loader import load_dataframe_smart
    path = Path(data_path)
    if not path.exists():
        return None
    return load_dataframe_smart(path, max_rows=max_rows, sample=sample)


def _convert_to_native_type(value: Any) -> Any:
    """Convert numpy/pandas types to native Python types for JSON serialization."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, pd.Int64Dtype)):
        return int(value)
    if isinstance(value, (np.floating, pd.Float64Dtype)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, (dict, list)):
        # Recursively convert nested structures
        if isinstance(value, dict):
            return {k: _convert_to_native_type(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_convert_to_native_type(v) for v in value]
    return value


# ---- Data Query Handlers ---------------------------------------------------

def _find_column_case_insensitive(df: pd.DataFrame, column: str) -> Optional[str]:
    """Find a column by name (case-insensitive)."""
    if column in df.columns:
        return column
    for c in df.columns:
        if c.lower() == column.lower():
            return c
    return None


def _coerce_filter_value(df: pd.DataFrame, column: str, value: Any) -> Any:
    """Try to coerce a filter value to match column dtype."""
    try:
        if pd.api.types.is_integer_dtype(df[column].dtype):
            return int(value)
        elif pd.api.types.is_float_dtype(df[column].dtype):
            return float(value)
    except (ValueError, TypeError):
        pass
    return value


def _query_find_columns(df: pd.DataFrame, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Find columns that match keywords."""
    keywords = arguments.get("keywords", [])
    if not keywords:
        return {"success": False, "error": "Missing keywords parameter"}
    
    matched_columns = []
    keyword_lower = [k.lower() for k in keywords]
    
    for col in df.columns:
        col_lower = col.lower()
        for keyword in keyword_lower:
            if keyword in col_lower or col_lower in keyword:
                matched_columns.append({"column": col, "matched_keyword": keyword})
                break
    
    if not matched_columns:
        return {
            "success": False,
            "error": f"No columns found matching keywords: {', '.join(keywords)}. Available columns: {', '.join(df.columns.tolist())}"
        }
    
    return {
        "success": True,
        "keywords": keywords,
        "matched_columns": matched_columns,
        "column_names": [c["column"] for c in matched_columns],
    }


def _query_search_rows(df: pd.DataFrame, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Search for rows containing a keyword (partial, case-insensitive match)."""
    column = arguments.get("column")
    keyword = arguments.get("keyword", "")
    limit = arguments.get("limit", 5)
    
    if not column:
        return {"success": False, "error": "Missing column parameter"}
    
    col_match = _find_column_case_insensitive(df, column)
    if not col_match:
        return {"success": False, "error": f"Column '{column}' not found. Available: {', '.join(df.columns[:10])}"}
    
    col_values = df[col_match].astype(str).str.lower()
    keyword_lower = keyword.lower()
    
    # Try multiple search strategies
    mask = col_values.str.contains(keyword_lower, na=False, regex=False)
    
    if mask.sum() == 0:
        keyword_nospace = keyword_lower.replace(" ", "")
        mask = col_values.str.contains(keyword_nospace, na=False, regex=False)
    
    if mask.sum() == 0:
        words = keyword_lower.split()
        if len(words) > 1:
            mask = pd.Series([True] * len(df), index=df.index)
            for word in words:
                mask = mask & col_values.str.contains(word, na=False, regex=False)
    
    matches = df[mask].head(limit)
    
    if len(matches) == 0:
        return {"success": False, "error": f"No rows found containing '{keyword}' in column '{col_match}'"}
    
    found_values = [_convert_to_native_type(v) for v in matches[col_match].tolist()]
    
    return {
        "success": True,
        "column": col_match,
        "keyword": keyword,
        "matches": found_values,
        "total_matches": int(mask.sum()),
        "showing": len(found_values),
    }


def _query_get_row(df: pd.DataFrame, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Get a row matching conditions."""
    column = arguments.get("column")
    value = arguments.get("value")
    
    if not column or value is None:
        return {"success": False, "error": "Missing column or value"}
    
    value = _coerce_filter_value(df, column, value)
    matches = df[df[column] == value]
    
    if len(matches) > 0:
        row_data = {k: _convert_to_native_type(v) for k, v in matches.iloc[0].to_dict().items()}
        return {"success": True, "data": row_data, "row_count": int(len(matches))}
    
    return {"success": False, "error": f"No rows found where {column} == {value}"}


def _query_get_value(df: pd.DataFrame, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Get a specific value from a row."""
    column = arguments.get("column")
    filter_column = arguments.get("filter_column")
    filter_value = arguments.get("filter_value")
    
    if not all([column, filter_column, filter_value]):
        return {"success": False, "error": "Missing required parameters"}
    
    filter_value = _coerce_filter_value(df, filter_column, filter_value)
    matches = df[df[filter_column] == filter_value]
    
    if len(matches) > 0:
        value = _convert_to_native_type(matches.iloc[0][column])
        return {"success": True, "value": value, "row_count": int(len(matches))}
    
    return {"success": False, "error": f"No rows found where {filter_column} == {filter_value}"}


def _query_calculate_ratio(df: pd.DataFrame, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate ratio between two columns."""
    numerator_col = arguments.get("numerator_column")
    denominator_col = arguments.get("denominator_column")
    filter_column = arguments.get("filter_column")
    filter_value = arguments.get("filter_value")
    
    if not numerator_col or not denominator_col:
        return {"success": False, "error": "Missing numerator or denominator column"}
    
    df_subset = df
    if filter_column and filter_value:
        filter_value = _coerce_filter_value(df, filter_column, filter_value)
        matches = df[df[filter_column] == filter_value]
        if len(matches) == 0:
            return {"success": False, "error": f"No rows found where {filter_column} == {filter_value}"}
        df_subset = matches
    
    numerator = df_subset[numerator_col].sum()
    denominator = df_subset[denominator_col].sum()
    
    if denominator == 0:
        return {"success": False, "error": "Denominator is zero"}
    
    return {
        "success": True,
        "numerator": _convert_to_native_type(numerator),
        "denominator": _convert_to_native_type(denominator),
        "ratio": _convert_to_native_type(numerator / denominator),
        "row_count": int(len(df_subset))
    }


def _apply_filter_to_df(df: pd.DataFrame, filter_column: str, filter_value: Any) -> tuple[pd.DataFrame, str, Optional[str]]:
    """Apply a filter to a DataFrame and return (filtered_df, filter_info, error)."""
    filter_col_match = _find_column_case_insensitive(df, filter_column)
    if not filter_col_match:
        return df, "", f"Filter column '{filter_column}' not found. Available: {', '.join(df.columns[:10])}"
    
    filter_value = _coerce_filter_value(df, filter_col_match, filter_value)
    
    # Try exact match first
    matches = df[df[filter_col_match].astype(str).str.lower() == str(filter_value).lower()]
    
    # If no exact match, try partial match
    if len(matches) == 0:
        matches = df[df[filter_col_match].astype(str).str.lower().str.contains(str(filter_value).lower(), na=False)]
    
    if len(matches) == 0:
        return df, "", f"No rows found where {filter_col_match} matches '{filter_value}'"
    
    filter_info = f" (filtered by {filter_col_match} = '{filter_value}', {len(matches)} rows)"
    return matches, filter_info, None


def _query_get_statistics(df: pd.DataFrame, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Get basic statistics with optional filtering."""
    column = arguments.get("column")
    filter_column = arguments.get("filter_column")
    filter_value = arguments.get("filter_value")
    
    if not column:
        return {"success": False, "error": "Missing column parameter"}
    
    col_match = _find_column_case_insensitive(df, column)
    if not col_match:
        return {"success": False, "error": f"Column '{column}' not found. Available: {', '.join(df.columns[:10])}"}
    
    df_subset = df
    filter_info = ""
    
    if filter_column and filter_value:
        df_subset, filter_info, error = _apply_filter_to_df(df, filter_column, filter_value)
        if error:
            return {"success": False, "error": error}
    
    if not pd.api.types.is_numeric_dtype(df_subset[col_match]):
        return {"success": False, "error": f"Column '{col_match}' is not numeric. Cannot calculate mean/variance/std/min/max."}
    
    stats = {
        "count": _convert_to_native_type(df_subset[col_match].count()),
        "nulls": _convert_to_native_type(df_subset[col_match].isna().sum()),
        "mean": _convert_to_native_type(df_subset[col_match].mean()),
        "variance": _convert_to_native_type(df_subset[col_match].var()),
        "std": _convert_to_native_type(df_subset[col_match].std()),
        "min": _convert_to_native_type(df_subset[col_match].min()),
        "max": _convert_to_native_type(df_subset[col_match].max()),
        "filter_info": filter_info if filter_info else None,
    }
    return {"success": True, "statistics": stats}


def _query_group_by(df: pd.DataFrame, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Group by a column and optionally aggregate."""
    column = arguments.get("column")
    agg_function = arguments.get("aggregation", "count")
    agg_column = arguments.get("aggregation_column")
    
    if not column:
        return {"success": False, "error": "Missing column parameter"}
    
    col_match = _find_column_case_insensitive(df, column)
    if not col_match:
        return {"success": False, "error": f"Column '{column}' not found"}
    
    try:
        grouped = df.groupby(col_match)
        
        if agg_function == "count":
            result = grouped.size().to_dict()
        elif agg_column:
            if agg_column not in df.columns:
                return {"success": False, "error": f"Aggregation column '{agg_column}' not found"}
            agg_map = {"sum": "sum", "mean": "mean", "min": "min", "max": "max"}
            if agg_function in agg_map:
                result = getattr(grouped[agg_column], agg_map[agg_function])().to_dict()
            else:
                result = grouped.size().to_dict()
        else:
            result = grouped.size().to_dict()
        
        result = {str(k): _convert_to_native_type(v) for k, v in result.items()}
        
        return {
            "success": True,
            "group_by": col_match,
            "aggregation": agg_function,
            "results": result,
            "total_groups": len(result)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _query_get_random_value(df: pd.DataFrame, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Get a random value from a column, or a random row."""
    column = arguments.get("column")
    
    if len(df) == 0:
        return {"success": False, "error": "Dataset is empty"}
    
    try:
        if column:
            col_match = _find_column_case_insensitive(df, column)
            if not col_match:
                return {"success": False, "error": f"Column '{column}' not found"}
            
            non_null_values = df[col_match].dropna()
            if len(non_null_values) == 0:
                return {"success": False, "error": f"Column '{col_match}' has no non-null values"}
            
            random_value = non_null_values.sample(n=1).iloc[0]
            return {"success": True, "column": col_match, "value": _convert_to_native_type(random_value)}
        else:
            random_row = df.sample(n=1).iloc[0]
            row_data = {k: _convert_to_native_type(v) for k, v in random_row.to_dict().items()}
            return {"success": True, "row": row_data}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Query handler dispatch table
_QUERY_HANDLERS: Dict[str, callable] = {
    "find_columns": _query_find_columns,
    "search_rows": _query_search_rows,
    "get_row": _query_get_row,
    "get_value": _query_get_value,
    "calculate_ratio": _query_calculate_ratio,
    "get_statistics": _query_get_statistics,
    "group_by": _query_group_by,
    "get_random_value": _query_get_random_value,
}


def _execute_data_query(df: pd.DataFrame, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a data query function on the DataFrame."""
    try:
        # Handle simple queries directly
        if function_name == "list_columns":
            return {"success": True, "columns": list(df.columns), "row_count": int(len(df))}
        
        if function_name == "get_row_count":
            return {"success": True, "row_count": int(len(df)), "column_count": int(len(df.columns))}
        
        # Dispatch to handler
        handler = _QUERY_HANDLERS.get(function_name)
        if handler:
            return handler(df, arguments)
        
        return {"success": False, "error": f"Unknown function: {function_name}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---- Query Execution Helpers -----------------------------------------------

def _execute_count_query(
    df: Optional[pd.DataFrame],
    column: str,
    value: str,
    is_large: bool,
    data_path: Optional[str],
) -> Optional[str]:
    """Execute a count query."""
    from data_loader import count_rows_chunked
    
    if is_large and data_path:
        count = count_rows_chunked(Path(data_path), column, value)
    elif df is not None:
        count = len(df[df[column].astype(str).str.lower() == str(value).lower()])
    else:
        return None
    return f"COUNT: There are **{count}** rows where {column} = '{value}'"


def _execute_stat_query(
    df: Optional[pd.DataFrame],
    column: str,
    stat: Optional[str],
    is_large: bool,
    data_path: Optional[str],
) -> Optional[str]:
    """Execute a statistics query."""
    from data_loader import calculate_stat_chunked
    
    stat = stat or "mean"
    
    if is_large and data_path:
        result = calculate_stat_chunked(Path(data_path), column, stat)
    elif df is not None:
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            return f"ERROR: Column '{column}' is not numeric"
        stat_funcs = {"mean": df[column].mean, "sum": df[column].sum, "min": df[column].min, "max": df[column].max}
        result = stat_funcs.get(stat, df[column].mean)()
    else:
        return None
    
    if result is not None:
        return f"STATISTIC: The {stat} of {column} is **{result:.2f}**"
    return f"ERROR: Could not calculate {stat} for {column}"


def _execute_group_count_query(
    df: Optional[pd.DataFrame],
    column: str,
    is_large: bool,
    data_path: Optional[str],
) -> Optional[str]:
    """Execute a group count query."""
    from data_loader import group_count_chunked
    
    if is_large and data_path:
        counts = group_count_chunked(Path(data_path), column)
    elif df is not None and column in df.columns:
        counts = df[column].value_counts().to_dict()
    else:
        return None
    
    result_str = "\n".join([f"- {k}: {v}" for k, v in counts.items()])
    return f"GROUP COUNT by {column}:\n{result_str}"


def _execute_random_query(
    df: Optional[pd.DataFrame],
    column: Optional[str],
    value: Optional[str],
    is_large: bool,
    data_path: Optional[str],
) -> Optional[str]:
    """Execute a random sample query."""
    from data_loader import get_random_sample_chunked
    
    if is_large and data_path:
        sample_df = get_random_sample_chunked(Path(data_path), column, value, n=1)
        if len(sample_df) > 0:
            sample = sample_df.iloc[0]
        else:
            return f"No rows found where {column} = '{value}'" if column and value else "No data available"
    elif df is not None:
        if column and value:
            filtered = df[df[column].astype(str).str.lower() == str(value).lower()]
            if len(filtered) > 0:
                sample = filtered.sample(n=1).iloc[0]
            else:
                return f"No rows found where {column} = '{value}'"
        else:
            sample = df.sample(n=1).iloc[0]
    else:
        return None
    
    return f"RANDOM SAMPLE:\n{sample.to_dict()}"


def _execute_search_query(
    df: Optional[pd.DataFrame],
    column: str,
    value: str,
    is_large: bool,
    data_path: Optional[str],
) -> Optional[str]:
    """Execute a search query."""
    search_df = df if df is not None else _load_dataframe(data_path, max_rows=LARGE_DATASET_SAMPLE_SIZE, sample=True)
    
    if search_df is None or column not in search_df.columns:
        return f"ERROR: Column '{column}' not found"
    
    matches = search_df[search_df[column].astype(str).str.lower().str.contains(str(value).lower(), na=False)]
    
    if len(matches) > 0:
        results = matches.head(5)[column].tolist()
        note = " (showing results from sample)" if is_large else ""
        return f"SEARCH RESULTS for '{value}' in {column}{note}:\n" + "\n".join([f"- {r}" for r in results])
    
    return f"No matches found for '{value}' in {column}"


def _build_query_plan_prompt(columns: List[str], row_count: int, user_message: str) -> str:
    """Build the LLM prompt for query planning."""
    return f"""Given this user question about a dataset, output a JSON query plan.

Dataset columns: {', '.join(columns)}
Dataset has {row_count} rows.

User question: {user_message}

Output ONLY valid JSON with one of these formats:

1. For counts/how many: {{"type": "count", "column": "col_name", "value": "filter_value"}}
2. For statistics (mean/avg/sum/min/max): {{"type": "stat", "stat": "mean|sum|min|max", "column": "col_name"}}
3. For group counts: {{"type": "group_count", "column": "col_name"}}
4. For random sample: {{"type": "random", "column": "col_name", "value": "optional_filter"}}
5. For search: {{"type": "search", "column": "col_name", "value": "search_term"}}
6. For general info: {{"type": "info"}}

Examples:
- "how many hard problems" -> {{"type": "count", "column": "Difficulty Level", "value": "Hard"}}
- "mean of success rate" -> {{"type": "stat", "stat": "mean", "column": "Success Rate"}}
- "give me any hard problem" -> {{"type": "random", "column": "Difficulty Level", "value": "Hard"}}
- "breakdown by difficulty" -> {{"type": "group_count", "column": "Difficulty Level"}}"""


def _find_column_in_list(column: str, columns: List[str]) -> str:
    """Find a column name in a list (case-insensitive)."""
    for c in columns:
        if c.lower() == column.lower():
            return c
    return column


async def _interpret_and_execute_query(
    df: Optional[pd.DataFrame],
    user_message: str,
    columns: List[str],
    data_path: Optional[str] = None,
) -> Optional[str]:
    """Use LLM to interpret the query and execute it efficiently (handles large datasets)."""
    from data_loader import is_large_dataset
    
    is_large = False
    if data_path:
        is_large, _ = is_large_dataset(Path(data_path))
    
    row_count = len(df) if df is not None else 0
    plan_prompt = _build_query_plan_prompt(columns, row_count, user_message)
    
    try:
        plan_response = await chat_completion([{"role": "user", "content": plan_prompt}], temperature=0, max_tokens=200)
        plan = _parse_json_response(plan_response)
        
        if not plan:
            return None
        
        query_type = plan.get("type", "info")
        column = plan.get("column")
        value = plan.get("value")
        stat = plan.get("stat")
        
        if column:
            column = _find_column_in_list(column, columns)
        
        # Dispatch to appropriate handler
        if query_type == "count" and column and value:
            return _execute_count_query(df, column, value, is_large, data_path)
        elif query_type == "stat" and column:
            return _execute_stat_query(df, column, stat, is_large, data_path)
        elif query_type == "group_count" and column:
            return _execute_group_count_query(df, column, is_large, data_path)
        elif query_type == "random":
            return _execute_random_query(df, column, value, is_large, data_path)
        elif query_type == "search" and column and value:
            return _execute_search_query(df, column, value, is_large, data_path)
        elif query_type == "info":
            info_note = " (large dataset - using estimates)" if is_large else ""
            return f"DATASET INFO{info_note}: {row_count} rows, {len(columns)} columns\nColumns: {', '.join(columns)}"
        
        return None
            
    except Exception as e:
        print(f"[Chat] Query interpretation failed: {e}")
        return None


async def _try_semantic_search(
    session: Optional["AsyncSession"],
    dataset_id: Optional[str],
    user_message: str,
) -> Optional[str]:
    """Try semantic search and return formatted results."""
    if not session or not dataset_id:
        return None
    
    try:
        from embeddings import semantic_search, has_embeddings
        if not await has_embeddings(session, dataset_id):
            return None
        
        print(f"[Chat] Using semantic search for: {user_message[:50]}...")
        results = await semantic_search(session, dataset_id, user_message, limit=SEARCH_RESULT_LIMIT)
        
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


def _get_sample_data_results(df: Optional[pd.DataFrame], data_path: Optional[str]) -> str:
    """Get sample data as a fallback query result."""
    if df is not None and len(df) > 0:
        sample_rows = df.head(3).to_dict(orient="records")
        return "SAMPLE DATA:\n" + "\n".join([str(row) for row in sample_rows])
    
    if data_path:
        sample_df = _load_dataframe(data_path, max_rows=3)
        if sample_df is not None and len(sample_df) > 0:
            sample_rows = sample_df.to_dict(orient="records")
            return "SAMPLE DATA:\n" + "\n".join([str(row) for row in sample_rows])
        return "DATASET INFO: Large dataset - queries will be processed efficiently using chunked operations."
    
    return "DATASET INFO: No data available."


def _build_data_response_prompt(stats: Dict[str, Any], query_results: str) -> str:
    """Build the system prompt for responding to data queries."""
    return f"""You are the Dataset Curator assistant. Answer the user's question using ONLY the data provided below.

Dataset info: {stats['rows']} rows, {stats['columns']} columns
Columns: {', '.join(stats['column_names'])}

QUERY RESULTS:
{query_results}

RULES:
1. Use ONLY the data shown above
2. Be direct and concise
3. Format numbers nicely"""


async def _chat_without_tools(
    df: Optional[pd.DataFrame],
    user_message: str,
    context: Optional[Dict[str, Any]] = None,
    session: Optional["AsyncSession"] = None,
    dataset_id: Optional[str] = None,
    data_path: Optional[str] = None,
) -> str:
    """Fallback chat without function calling - uses LLM to interpret and execute queries."""
    columns = context.get("columns", []) if context else (list(df.columns) if df is not None else [])
    
    stats = {
        "rows": len(df) if df is not None else 0,
        "columns": len(df.columns) if df is not None else len(columns),
        "column_names": columns[:15],
    }
    
    # Try multiple query strategies in order
    query_results = await _interpret_and_execute_query(df, user_message, columns, data_path)
    
    if not query_results:
        query_results = await _try_semantic_search(session, dataset_id, user_message)
    
    if not query_results:
        query_results = _get_sample_data_results(df, data_path)
    
    system_prompt = _build_data_response_prompt(stats, query_results)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    return await chat_completion(messages, temperature=0.3)


async def _classify_query_type(user_message: str, columns: List[str]) -> str:
    """Use LLM to classify if query is about the dataset or general conversation."""
    
    columns_str = ", ".join(columns[:15]) if columns else "unknown"
    
    system_prompt = f"""You are a query classifier. Determine if the user's message is:
1. "data" - A question or request about the loaded dataset (wants to query, search, filter, get statistics, find records, etc.)
2. "general" - General conversation, greetings, thanks, questions about you, or unrelated topics

The user has a dataset loaded with columns: {columns_str}

Respond with ONLY "data" or "general" - nothing else.

Examples:
- "How are you?" → general
- "Hello" → general
- "Thanks!" → general
- "What can you do?" → general
- "Give me any hard problem" → data
- "Find similar questions for two sum" → data
- "How many rows are there?" → data
- "Show me a random record" → data
- "What is the average success rate?" → data
- "List all columns" → data
- "What is this data about?" → data
- "Filter by difficulty" → data"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    response = await chat_completion(messages, temperature=0, max_tokens=10)
    result = response.strip().lower()
    return "data" if "data" in result else "general"


def _prepare_chat_context(
    data_path: Optional[str],
    context: Optional[Dict[str, Any]],
) -> tuple[Optional[pd.DataFrame], List[str], bool]:
    """Prepare DataFrame and columns for chat, handling large datasets."""
    from data_loader import is_large_dataset
    
    is_large = False
    if data_path:
        is_large, _ = is_large_dataset(Path(data_path))
    
    df = _load_dataframe(data_path, max_rows=LARGE_DATASET_SAMPLE_SIZE if is_large else None, sample=is_large)
    columns = context.get("columns", []) if context else []
    if df is not None and not columns:
        columns = list(df.columns)
    
    return df, columns, is_large


async def _handle_general_conversation(
    user_message: str,
    history: Optional[List[Dict[str, str]]],
) -> str:
    """Handle general conversation (non-data queries)."""
    general_prompt = """You are the Dataset Curator assistant, a friendly AI that helps users work with datasets.
You're currently helping a user who has data loaded. Be conversational and helpful.
If they ask what you can do, mention: searching data, getting statistics, finding specific records, filtering, etc."""
    
    messages: List[Message] = [{"role": "system", "content": general_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return await chat_completion(messages, temperature=0.7)


async def _execute_tool_calls(
    df: pd.DataFrame,
    message: Any,
    messages: List[Message],
    client: AsyncOpenAI,
) -> tuple[Any, List[Message]]:
    """Execute tool calls and return updated message and messages list."""
    iteration = 0
    
    while message.tool_calls and iteration < MAX_TOOL_ITERATIONS:
        iteration += 1
        messages.append(message)
        
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}
            
            print(f"[Chat] Calling function: {function_name}({arguments})")
            result = _execute_data_query(df, function_name, arguments)
            print(f"[Chat] Result: {result}")
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
        
        response = await client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            tools=CHAT_TOOLS,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=2048,
            stream=False,
        )
        message = response.choices[0].message
    
    return message, messages


async def _handle_data_query_with_tools(
    df: pd.DataFrame,
    user_message: str,
    columns: List[str],
    history: Optional[List[Dict[str, str]]],
    context: Optional[Dict[str, Any]],
    session: Optional["AsyncSession"],
    dataset_id: Optional[str],
    data_path: Optional[str],
) -> str:
    """Handle data queries using function calling."""
    system_prompt = _build_chat_system_prompt(columns)
    messages: List[Message] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    
    try:
        client = get_client()
        response = await client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            tools=CHAT_TOOLS,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=2048,
            stream=False,
        )
        
        message = response.choices[0].message
        print(f"[Chat] Initial response: content={repr(message.content)}, tool_calls={message.tool_calls}")
        
        if not message.content and not message.tool_calls:
            print("[Chat] Empty response with no tools, retrying without function calling...")
            return await _chat_without_tools(df, user_message, context, session, dataset_id, data_path)
        
        message, messages = await _execute_tool_calls(df, message, messages, client)
        
        if message.content:
            return message.content
        
        print(f"[Chat] No content after tool calls, falling back...")
        return await _chat_without_tools(df, user_message, context, session, dataset_id, data_path)
            
    except Exception as e:
        error_str = str(e).lower()
        print(f"[Chat] Error: {e}")
        
        if any(indicator in error_str for indicator in ["tool", "function", "no endpoints found"]):
            print("[Chat] Tool calling not supported, falling back to non-tool chat...")
            return await _chat_without_tools(df, user_message, context, session, dataset_id, data_path)
        
        raise


async def chat_with_agent(
    user_message: str,
    data_path: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    session: Optional["AsyncSession"] = None,
    dataset_id: Optional[str] = None,
) -> str:
    """Chat with the dataset curator agent - answers questions about data using function calling."""
    df, columns, _ = _prepare_chat_context(data_path, context)
    
    query_type = await _classify_query_type(user_message, columns)
    print(f"[Chat] Query: {user_message[:50]}... | Type: {query_type}")
    
    if query_type == "general":
        return await _handle_general_conversation(user_message, history)
    
    if df is not None:
        return await _handle_data_query_with_tools(
            df, user_message, columns, history, context, session, dataset_id, data_path
        )
    
    # No data available, use regular chat
    system_prompt = _build_chat_system_prompt(columns)
    messages: List[Message] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return await chat_completion(messages, temperature=0.7)
