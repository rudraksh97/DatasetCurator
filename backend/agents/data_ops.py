"""Natural language data operations executor.

This module provides the DataOperator class for executing data transformations
on pandas DataFrames based on parsed operation intents from the LLM.

Supported operations include:
- Column operations: drop, rename, add, keep
- Row operations: filter, drop, drop_nulls, drop_duplicates
- Data transformations: type conversion, case changes, value replacement
- Conditional columns: add columns based on conditions
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd


class DataOperator:
    """Execute data transformations based on parsed intents."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        self.operations_log: List[str] = []

    # ---- Internal helpers -------------------------------------------------

    def _match_column(self, col: str) -> Optional[str]:
        """Return the best matching column (case-insensitive, partial)."""
        if col in self.df.columns:
            return col
        lower = col.lower()
        # Exact case-insensitive
        for c in self.df.columns:
            if c.lower() == lower:
                return c
        # Partial match
        for c in self.df.columns:
            if lower in c.lower() or c.lower() in lower:
                return c
        return None

    def _coerce_value(self, col: str, value: Any) -> Any:
        """Try to coerce value to match column dtype."""
        try:
            col_dtype = self.df[col].dtype
            if pd.api.types.is_integer_dtype(col_dtype):
                return int(value)
            if pd.api.types.is_float_dtype(col_dtype):
                return float(value)
        except Exception:
            pass
        return value

    def execute(self, operation: str, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute a single operation and return success status and message."""
        op_map = {
            "drop_column": self._drop_column,
            "drop_columns": self._drop_columns,
            "rename_column": self._rename_column,
            "drop_nulls": self._drop_nulls,
            "fill_nulls": self._fill_nulls,
            "drop_duplicates": self._drop_duplicates,
            "filter_rows": self._filter_rows,
            "drop_rows": self._drop_rows,
            "convert_type": self._convert_type,
            "strip_whitespace": self._strip_whitespace,
            "lowercase": self._lowercase,
            "uppercase": self._uppercase,
            "replace_values": self._replace_values,
            "sort": self._sort,
            "keep_columns": self._keep_columns,
            "add_column": self._add_column,
            "add_conditional_column": self._add_conditional_column,
        }
        
        if operation not in op_map:
            return False, f"Unknown operation: {operation}"
        
        try:
            result = op_map[operation](params)
            self.operations_log.append(f"{operation}: {params}")
            return True, result
        except Exception as e:
            return False, f"Error executing {operation}: {str(e)}"

    def _drop_column(self, params: Dict) -> str:
        col = params.get("column")
        col = self._match_column(col) if col else None
        if not col:
            return f"Column not found. Available: {list(self.df.columns)}"
        self.df = self.df.drop(columns=[col])
        return f"Dropped column '{col}'. Now {len(self.df.columns)} columns."

    def _drop_columns(self, params: Dict) -> str:
        cols = params.get("columns", [])
        dropped = []
        for col in cols:
            if col in self.df.columns:
                self.df = self.df.drop(columns=[col])
                dropped.append(col)
        return f"Dropped {len(dropped)} columns: {dropped}"

    def _rename_column(self, params: Dict) -> str:
        old_name = params.get("old_name")
        new_name = params.get("new_name")
        if old_name not in self.df.columns:
            return f"Column '{old_name}' not found."
        self.df = self.df.rename(columns={old_name: new_name})
        return f"Renamed '{old_name}' to '{new_name}'."

    def _drop_nulls(self, params: Dict) -> str:
        col = params.get("column")
        before = len(self.df)
        if col:
            if col not in self.df.columns:
                return f"Column '{col}' not found."
            self.df = self.df.dropna(subset=[col])
        else:
            self.df = self.df.dropna()
        after = len(self.df)
        return f"Removed {before - after} rows with null values. Now {after} rows."

    def _fill_nulls(self, params: Dict) -> str:
        col = params.get("column")
        value = params.get("value")
        method = params.get("method")  # 'mean', 'median', 'mode', 'ffill', 'bfill'
        
        if col and col not in self.df.columns:
            return f"Column '{col}' not found."
        
        if method:
            if method == "mean" and col:
                value = self.df[col].mean()
            elif method == "median" and col:
                value = self.df[col].median()
            elif method == "mode" and col:
                value = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else None
            elif method == "ffill":
                if col:
                    self.df[col] = self.df[col].ffill()
                else:
                    self.df = self.df.ffill()
                return f"Filled nulls using forward fill."
            elif method == "bfill":
                if col:
                    self.df[col] = self.df[col].bfill()
                else:
                    self.df = self.df.bfill()
                return f"Filled nulls using backward fill."
        
        if col:
            count = self.df[col].isna().sum()
            self.df[col] = self.df[col].fillna(value)
            return f"Filled {count} null values in '{col}' with '{value}'."
        else:
            self.df = self.df.fillna(value)
            return f"Filled all null values with '{value}'."

    def _drop_duplicates(self, params: Dict) -> str:
        cols = params.get("columns")
        keep = params.get("keep", "first")
        before = len(self.df)
        if cols:
            self.df = self.df.drop_duplicates(subset=cols, keep=keep)
        else:
            self.df = self.df.drop_duplicates(keep=keep)
        after = len(self.df)
        return f"Removed {before - after} duplicate rows. Now {after} rows."

    def _filter_rows(self, params: Dict) -> str:
        col = params.get("column")
        operator = params.get("operator", "==")
        value = params.get("value")

        col = self._match_column(col) if col else None
        if not col:
            return f"Column not found. Available: {list(self.df.columns)}"

        before = len(self.df)
        
        # Try to convert value to match column dtype
        value = self._coerce_value(col, value)
        
        if operator == "==":
            self.df = self.df[self.df[col] == value]
        elif operator == "!=":
            self.df = self.df[self.df[col] != value]
        elif operator == ">":
            self.df = self.df[self.df[col] > value]
        elif operator == "<":
            self.df = self.df[self.df[col] < value]
        elif operator == ">=":
            self.df = self.df[self.df[col] >= value]
        elif operator == "<=":
            self.df = self.df[self.df[col] <= value]
        elif operator == "contains":
            self.df = self.df[self.df[col].astype(str).str.contains(str(value), case=False, na=False)]
        
        after = len(self.df)
        return f"Filtered to {after} rows (removed {before - after})."

    def _drop_rows(self, params: Dict) -> str:
        col = params.get("column")
        operator = params.get("operator", "==")
        value = params.get("value")

        col = self._match_column(col) if col else None
        if not col:
            return f"Column not found. Available: {list(self.df.columns)}"

        before = len(self.df)
        
        # Try to convert value to match column dtype
        value = self._coerce_value(col, value)
        
        if operator == "==":
            self.df = self.df[self.df[col] != value]
        elif operator == "contains":
            self.df = self.df[~self.df[col].astype(str).str.contains(str(value), case=False, na=False)]
        
        after = len(self.df)
        return f"Dropped {before - after} rows. Now {after} rows."

    def _convert_type(self, params: Dict) -> str:
        col = params.get("column")
        dtype = params.get("dtype")
        
        if col not in self.df.columns:
            return f"Column '{col}' not found."
        
        type_map = {
            "int": "int64",
            "integer": "int64",
            "float": "float64",
            "number": "float64",
            "string": "str",
            "text": "str",
            "date": "datetime64[ns]",
            "datetime": "datetime64[ns]",
            "bool": "bool",
            "boolean": "bool",
        }
        dtype = type_map.get(dtype.lower(), dtype)
        
        try:
            if dtype == "datetime64[ns]":
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
            else:
                self.df[col] = self.df[col].astype(dtype)
            return f"Converted '{col}' to {dtype}."
        except Exception as e:
            return f"Error: Could not convert '{col}' to {dtype}: {str(e)}"

    def _strip_whitespace(self, params: Dict) -> str:
        col = params.get("column")
        if col:
            if col not in self.df.columns:
                return f"Column '{col}' not found."
            if self.df[col].dtype == "object":
                self.df[col] = self.df[col].str.strip()
            return f"Stripped whitespace from '{col}'."
        else:
            for c in self.df.select_dtypes(include=["object"]).columns:
                self.df[c] = self.df[c].str.strip()
            return "Stripped whitespace from all text columns."

    def _lowercase(self, params: Dict) -> str:
        col = params.get("column")
        if col not in self.df.columns:
            return f"Column '{col}' not found."
        self.df[col] = self.df[col].astype(str).str.lower()
        return f"Converted '{col}' to lowercase."

    def _uppercase(self, params: Dict) -> str:
        col = params.get("column")
        if col not in self.df.columns:
            return f"Column '{col}' not found."
        self.df[col] = self.df[col].astype(str).str.upper()
        return f"Converted '{col}' to uppercase."

    def _replace_values(self, params: Dict) -> str:
        col = params.get("column")
        old_value = params.get("old_value")
        new_value = params.get("new_value")
        
        if col:
            if col not in self.df.columns:
                return f"Column '{col}' not found."
            count = (self.df[col] == old_value).sum()
            self.df[col] = self.df[col].replace(old_value, new_value)
            return f"Replaced {count} occurrences of '{old_value}' with '{new_value}' in '{col}'."
        else:
            self.df = self.df.replace(old_value, new_value)
            return f"Replaced all '{old_value}' with '{new_value}'."

    def _sort(self, params: Dict) -> str:
        col = params.get("column")
        ascending = params.get("ascending", True)
        
        if col not in self.df.columns:
            return f"Column '{col}' not found."
        
        self.df = self.df.sort_values(by=col, ascending=ascending)
        order = "ascending" if ascending else "descending"
        return f"Sorted by '{col}' ({order})."

    def _keep_columns(self, params: Dict) -> str:
        cols = params.get("columns", [])
        valid_cols = [c for c in cols if c in self.df.columns]
        if not valid_cols:
            return f"None of the specified columns found."
        self.df = self.df[valid_cols]
        return f"Kept {len(valid_cols)} columns: {valid_cols}"

    def _add_column(self, params: Dict) -> str:
        name = params.get("name")
        value = params.get("value")
        from_column = params.get("from_column")
        operation = params.get("operation")
        
        if not name:
            return "Column name is required."
        
        if from_column and from_column in self.df.columns:
            if operation == "copy":
                self.df[name] = self.df[from_column]
            elif operation == "upper":
                self.df[name] = self.df[from_column].astype(str).str.upper()
            elif operation == "lower":
                self.df[name] = self.df[from_column].astype(str).str.lower()
            elif operation == "length":
                self.df[name] = self.df[from_column].astype(str).str.len()
            else:
                self.df[name] = self.df[from_column]
            return f"Added column '{name}' based on '{from_column}'."
        elif value is not None:
            self.df[name] = value
            return f"Added column '{name}' with value '{value}'."
        else:
            self.df[name] = None
            return f"Added empty column '{name}'."

    def _resolve_condition_column(self, condition_col: str) -> Optional[str]:
        """Resolve condition column name (case-insensitive)."""
        if condition_col in self.df.columns:
            return condition_col
        matches = [c for c in self.df.columns if c.lower() == condition_col.lower()]
        return matches[0] if matches else None

    def _parse_conditions_param(self, conditions: Any) -> Optional[List[Dict]]:
        """Parse conditions parameter (handles string JSON)."""
        if isinstance(conditions, str):
            try:
                import json
                return json.loads(conditions)
            except:
                return None
        return conditions if isinstance(conditions, list) else None

    def _build_condition_mask(self, condition_col: str, cond: Dict) -> Optional[Any]:
        """Build a condition mask for a single condition."""
        op = cond.get("operator", ">")
        val = cond.get("value")
        
        operator_map = {
            "<": lambda: self.df[condition_col] < val,
            "<=": lambda: self.df[condition_col] <= val,
            ">": lambda: self.df[condition_col] > val,
            ">=": lambda: self.df[condition_col] >= val,
            "==": lambda: self.df[condition_col] == val,
            "!=": lambda: self.df[condition_col] != val,
        }
        
        if op in operator_map:
            return operator_map[op]()
        
        if op in ("between", "range"):
            val1 = cond.get("value1") or cond.get("min")
            val2 = cond.get("value2") or cond.get("max")
            if val1 is not None and val2 is not None:
                return (self.df[condition_col] >= val1) & (self.df[condition_col] <= val2)
        
        return None

    def _apply_multi_conditions(self, name: str, condition_col: str, conditions: List[Dict], default_value: Any) -> str:
        """Apply multiple conditions using np.select."""
        cond_list = []
        choice_list = []
        
        for cond in conditions:
            mask = self._build_condition_mask(condition_col, cond)
            if mask is not None:
                cond_list.append(mask)
                choice_list.append(cond.get("result"))
        
        if not cond_list:
            return "Error: No valid conditions provided."
        
        self.df[name] = np.select(cond_list, choice_list, default=default_value)
        return f"Added column '{name}' with {len(conditions)} conditional ranges."

    def _apply_single_condition(self, name: str, condition_col: str, operator: str, threshold: Any, true_value: Any, false_value: Any) -> str:
        """Apply a single condition to create a boolean column."""
        operator_map = {
            ">": lambda: self.df[condition_col] > threshold,
            "<": lambda: self.df[condition_col] < threshold,
            ">=": lambda: self.df[condition_col] >= threshold,
            "<=": lambda: self.df[condition_col] <= threshold,
            "==": lambda: self.df[condition_col] == threshold,
            "!=": lambda: self.df[condition_col] != threshold,
            "contains": lambda: self.df[condition_col].astype(str).str.contains(str(threshold), case=False, na=False),
        }
        
        if operator in operator_map:
            self.df[name] = operator_map[operator]()
        else:
            self.df[name] = False
        
        # Apply custom true/false values if not boolean
        if true_value is not True or false_value is not False:
            self.df[name] = self.df[name].map({True: true_value, False: false_value})
        
        return f"Added column '{name}' based on condition: {condition_col} {operator} {threshold}."

    def _add_conditional_column(self, params: Dict) -> str:
        name = params.get("name")
        condition_col = params.get("condition_column")
        
        if not name or not condition_col:
            return "Column name and condition column are required."
        
        resolved_col = self._resolve_condition_column(condition_col)
        if not resolved_col:
            return f"Column '{condition_col}' not found."
        condition_col = resolved_col
        
        try:
            conditions = self._parse_conditions_param(params.get("conditions"))
            
            if conditions:
                default_value = params.get("default_value", "Unknown")
                return self._apply_multi_conditions(name, condition_col, conditions, default_value)
            
            # Single condition
            operator = params.get("operator", ">")
            threshold = params.get("threshold")
            true_value = params.get("true_value", True)
            false_value = params.get("false_value", False)
            
            return self._apply_single_condition(name, condition_col, operator, threshold, true_value, false_value)
        except Exception as e:
            return f"Error: Error creating conditional column: {str(e)}"

    def get_result(self) -> pd.DataFrame:
        """Return the transformed DataFrame."""
        return self.df

    def get_summary(self) -> str:
        """Get a summary of changes made.
        
        Returns:
            Formatted string summarizing row/column changes and operations performed.
        """
        orig_rows, orig_cols = self.original_shape
        new_rows, new_cols = self.df.shape
        
        summary = f"**Changes Applied:**\n"
        summary += f"- Rows: {orig_rows} → {new_rows} ({new_rows - orig_rows:+d})\n"
        summary += f"- Columns: {orig_cols} → {new_cols} ({new_cols - orig_cols:+d})\n"
        
        if self.operations_log:
            summary += f"\n**Operations ({len(self.operations_log)}):**\n"
            for op in self.operations_log[-5:]:
                summary += f"- {op}\n"
        
        return summary
