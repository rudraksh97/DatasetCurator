"""LLM helpers for intent classification and data-aware chat."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

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
    
    try:
        response = await chat_completion(messages, temperature=0.1, max_tokens=1000)
        result = _parse_json_response(response)
        if result and "steps" in result:
            return result
    except Exception as e:
        print(f"Planning error: {e}")
    
    # Fallback: return empty plan
    return {"is_multi_step": False, "steps": []}

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

"transform_data": {{"operation": "op_name", ...}}

Examples:
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

Available functions:
- get_row: Get a row matching a condition (column == value)
- get_value: Get a specific value from a row (filter by one column, get another column's value)
- get_random_value: Get a random value from a column, or a random row (use for "random X", "give me a random Y" questions)
- calculate_ratio: Calculate ratio between two columns (optionally filtered)
- get_statistics: Get statistics for a column (count, nulls, mean, variance, std, min, max)
- group_by: Group data by a column and count/aggregate (use for "breakdown", "group by", "count by" questions)
- list_columns: List all columns in the dataset
- get_row_count: Get total row and column count

When answering questions:
1. First, understand what data is needed
2. Call the appropriate function(s) to get the actual data
3. Use the returned values in your answer
4. Show your calculations if doing math

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
            "name": "get_row",
            "description": "Get a row from the dataset matching a condition. Use this to find a specific record.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name to filter by"},
                    "value": {"type": "string", "description": "Value to match (will be auto-converted to match column type)"},
                },
                "required": ["column", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_value",
            "description": "Get a specific value from a row. Filter by one column, get another column's value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name to get the value from"},
                    "filter_column": {"type": "string", "description": "Column name to filter by"},
                    "filter_value": {"type": "string", "description": "Value to filter by (will be auto-converted)"},
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
            "description": "Get statistics for a column (count, nulls, mean, variance, std, min, max)",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name"}
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
    
    try:
        response = await chat_completion(messages, temperature=0.1, max_tokens=500)
        result = _parse_json_response(response)
        if result and "intent" in result:
            return result
    except Exception as e:
        print(f"Intent classification error: {e}")
    
    return {"intent": "chat", "params": {}, "explanation": "Could not classify intent"}


def _load_dataframe(data_path: Optional[str]) -> Optional[pd.DataFrame]:
    """Load DataFrame from path."""
    if not data_path:
        return None
    try:
        path = Path(data_path)
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return None


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


def _execute_data_query(df: pd.DataFrame, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a data query function on the DataFrame."""
    try:
        if function_name == "get_row":
            # Get a row matching conditions
            column = arguments.get("column")
            value = arguments.get("value")
            if column and value:
                # Try type conversion
                try:
                    if pd.api.types.is_integer_dtype(df[column].dtype):
                        value = int(value)
                    elif pd.api.types.is_float_dtype(df[column].dtype):
                        value = float(value)
                except (ValueError, TypeError):
                    pass
                
                matches = df[df[column] == value]
                if len(matches) > 0:
                    row_data = matches.iloc[0].to_dict()
                    # Convert all values to native types
                    row_data = {k: _convert_to_native_type(v) for k, v in row_data.items()}
                    return {"success": True, "data": row_data, "row_count": int(len(matches))}
                return {"success": False, "error": f"No rows found where {column} == {value}"}
            return {"success": False, "error": "Missing column or value"}
        
        elif function_name == "get_value":
            # Get a specific value from a row
            column = arguments.get("column")
            filter_column = arguments.get("filter_column")
            filter_value = arguments.get("filter_value")
            if column and filter_column and filter_value:
                # Try type conversion
                try:
                    if pd.api.types.is_integer_dtype(df[filter_column].dtype):
                        filter_value = int(filter_value)
                    elif pd.api.types.is_float_dtype(df[filter_column].dtype):
                        filter_value = float(filter_value)
                except (ValueError, TypeError):
                    pass
                
                matches = df[df[filter_column] == filter_value]
                if len(matches) > 0:
                    value = matches.iloc[0][column]
                    value = _convert_to_native_type(value)
                    return {"success": True, "value": value, "row_count": int(len(matches))}
                return {"success": False, "error": f"No rows found where {filter_column} == {filter_value}"}
            return {"success": False, "error": "Missing required parameters"}
        
        elif function_name == "calculate_ratio":
            # Calculate ratio between two columns
            numerator_col = arguments.get("numerator_column")
            denominator_col = arguments.get("denominator_column")
            filter_column = arguments.get("filter_column")
            filter_value = arguments.get("filter_value")
            
            if numerator_col and denominator_col:
                if filter_column and filter_value:
                    # Filter first
                    try:
                        if pd.api.types.is_integer_dtype(df[filter_column].dtype):
                            filter_value = int(filter_value)
                        elif pd.api.types.is_float_dtype(df[filter_column].dtype):
                            filter_value = float(filter_value)
                    except (ValueError, TypeError):
                        pass
                    matches = df[df[filter_column] == filter_value]
                    if len(matches) == 0:
                        return {"success": False, "error": f"No rows found where {filter_column} == {filter_value}"}
                    df_subset = matches
                else:
                    df_subset = df
                
                numerator = df_subset[numerator_col].sum()
                denominator = df_subset[denominator_col].sum()
                if denominator == 0:
                    return {"success": False, "error": "Denominator is zero"}
                ratio = numerator / denominator
                return {
                    "success": True,
                    "numerator": _convert_to_native_type(numerator),
                    "denominator": _convert_to_native_type(denominator),
                    "ratio": _convert_to_native_type(ratio),
                    "row_count": int(len(df_subset))
                }
            return {"success": False, "error": "Missing numerator or denominator column"}
        
        elif function_name == "get_statistics":
            # Get basic statistics
            column = arguments.get("column")
            if column:
                if column not in df.columns:
                    return {"success": False, "error": f"Column '{column}' not found"}
                stats = {
                    "count": _convert_to_native_type(df[column].count()),
                    "nulls": _convert_to_native_type(df[column].isna().sum()),
                    "mean": _convert_to_native_type(df[column].mean()) if pd.api.types.is_numeric_dtype(df[column]) else None,
                    "variance": _convert_to_native_type(df[column].var()) if pd.api.types.is_numeric_dtype(df[column]) else None,
                    "std": _convert_to_native_type(df[column].std()) if pd.api.types.is_numeric_dtype(df[column]) else None,
                    "min": _convert_to_native_type(df[column].min()) if pd.api.types.is_numeric_dtype(df[column]) else None,
                    "max": _convert_to_native_type(df[column].max()) if pd.api.types.is_numeric_dtype(df[column]) else None,
                }
                return {"success": True, "statistics": stats}
            return {"success": False, "error": "Missing column parameter"}
        
        elif function_name == "group_by":
            # Group by a column and optionally aggregate
            column = arguments.get("column")
            agg_function = arguments.get("aggregation", "count")  # count, sum, mean, etc.
            agg_column = arguments.get("aggregation_column")  # Optional: column to aggregate
            
            if not column:
                return {"success": False, "error": "Missing column parameter"}
            
            if column not in df.columns:
                # Try case-insensitive match
                matches = [c for c in df.columns if c.lower() == column.lower()]
                if matches:
                    column = matches[0]
                else:
                    return {"success": False, "error": f"Column '{column}' not found"}
            
            try:
                grouped = df.groupby(column)
                
                if agg_function == "count":
                    result = grouped.size().to_dict()
                elif agg_function == "sum" and agg_column:
                    if agg_column not in df.columns:
                        return {"success": False, "error": f"Aggregation column '{agg_column}' not found"}
                    result = grouped[agg_column].sum().to_dict()
                elif agg_function == "mean" and agg_column:
                    if agg_column not in df.columns:
                        return {"success": False, "error": f"Aggregation column '{agg_column}' not found"}
                    result = grouped[agg_column].mean().to_dict()
                elif agg_function == "min" and agg_column:
                    if agg_column not in df.columns:
                        return {"success": False, "error": f"Aggregation column '{agg_column}' not found"}
                    result = grouped[agg_column].min().to_dict()
                elif agg_function == "max" and agg_column:
                    if agg_column not in df.columns:
                        return {"success": False, "error": f"Aggregation column '{agg_column}' not found"}
                    result = grouped[agg_column].max().to_dict()
                else:
                    result = grouped.size().to_dict()
                
                # Convert to native types
                result = {str(k): _convert_to_native_type(v) for k, v in result.items()}
                
                return {
                    "success": True,
                    "group_by": column,
                    "aggregation": agg_function,
                    "results": result,
                    "total_groups": len(result)
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        elif function_name == "get_random_value":
            # Get a random value from a column, or a random row
            column = arguments.get("column")
            
            if len(df) == 0:
                return {"success": False, "error": "Dataset is empty"}
            
            try:
                if column:
                    # Get random value from specific column
                    if column not in df.columns:
                        # Try case-insensitive match
                        matches = [c for c in df.columns if c.lower() == column.lower()]
                        if matches:
                            column = matches[0]
                        else:
                            return {"success": False, "error": f"Column '{column}' not found"}
                    
                    # Filter out null values
                    non_null_values = df[column].dropna()
                    if len(non_null_values) == 0:
                        return {"success": False, "error": f"Column '{column}' has no non-null values"}
                    
                    random_value = non_null_values.sample(n=1).iloc[0]
                    return {
                        "success": True,
                        "column": column,
                        "value": _convert_to_native_type(random_value)
                    }
                else:
                    # Get random row
                    random_row = df.sample(n=1).iloc[0]
                    row_data = random_row.to_dict()
                    row_data = {k: _convert_to_native_type(v) for k, v in row_data.items()}
                    return {
                        "success": True,
                        "row": row_data
                    }
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        elif function_name == "list_columns":
            return {"success": True, "columns": list(df.columns), "row_count": int(len(df))}
        
        elif function_name == "get_row_count":
            return {"success": True, "row_count": int(len(df)), "column_count": int(len(df.columns))}
        
        return {"success": False, "error": f"Unknown function: {function_name}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def chat_with_agent(
    user_message: str,
    data_path: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Chat with the dataset curator agent - answers questions about data using function calling."""
    
    df = _load_dataframe(data_path)
    system_prompt = _build_chat_system_prompt(context.get("columns") if context else None)

    messages: List[Message] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    
    if not df is None:
        # Use function calling
        try:
            client = get_client()
            # Function calling requires non-streaming mode - explicitly set to False
            # Some providers may ignore this, so we catch the error
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
            
            # Handle function calls
            while message.tool_calls:
                messages.append(message)
                
                # Execute function calls
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    result = _execute_data_query(df, function_name, arguments)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
                
                # Get next response
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
            
            return message.content or "I couldn't process your request."
        except Exception as e:
            error_str = str(e)
            # If streaming mode error, the model might not support function calling
            if "streaming mode" in error_str.lower() or "tools are not supported" in error_str.lower():
                # Fallback: try without tools but with data context
                context_info = f"\n\nDataset has {len(df)} rows and {len(df.columns)} columns: {', '.join(df.columns.tolist()[:10])}"
                if len(df.columns) > 10:
                    context_info += f" and {len(df.columns) - 10} more."
                
                fallback_messages = messages + [{"role": "user", "content": user_message + context_info}]
                return await chat_completion(fallback_messages, temperature=0.7)
            raise
    else:
        # No data available, use regular chat
        return await chat_completion(messages, temperature=0.7)
