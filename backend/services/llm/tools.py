"""Tool definitions for LLM function calling.

This module defines the tools (functions) available to the LLM
for querying and interacting with datasets.

Query tools are organized by operation type:
- Column Discovery: find_columns, list_columns
- Row Lookup: search_rows, get_row, get_value, get_random_value
- Aggregation: get_statistics, group_by, calculate_ratio
- Metadata: get_row_count
"""
from __future__ import annotations

from typing import Any, Dict, List

# =============================================================================
# LLM Function Calling Tools (Query Operations - Read Only)
# =============================================================================

CHAT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "find_columns",
            "description": "Find columns in the dataset that match given keywords. USE THIS FIRST when the user mentions column names in their query (e.g., 'study hours', 'degree', 'bsc'). This helps identify which columns to use for filtering and calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of keywords to search for in column names (e.g., ['study hours', 'degree', 'bsc'])",
                    },
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
                    "column": {"type": "string", "description": "Optional: Column name to get a random value from. If not provided, returns a random row."},
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
