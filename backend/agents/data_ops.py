"""Natural language data operations executor."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd


class DataOperator:
    """Execute data transformations based on parsed intents."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        self.operations_log: List[str] = []

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
        if col not in self.df.columns:
            # Try case-insensitive match
            matches = [c for c in self.df.columns if c.lower() == col.lower()]
            if matches:
                col = matches[0]
            else:
                return f"Column '{col}' not found. Available: {list(self.df.columns)}"
        self.df = self.df.drop(columns=[col])
        return f"✅ Dropped column '{col}'. Now {len(self.df.columns)} columns."

    def _drop_columns(self, params: Dict) -> str:
        cols = params.get("columns", [])
        dropped = []
        for col in cols:
            if col in self.df.columns:
                self.df = self.df.drop(columns=[col])
                dropped.append(col)
        return f"✅ Dropped {len(dropped)} columns: {dropped}"

    def _rename_column(self, params: Dict) -> str:
        old_name = params.get("old_name")
        new_name = params.get("new_name")
        if old_name not in self.df.columns:
            return f"Column '{old_name}' not found."
        self.df = self.df.rename(columns={old_name: new_name})
        return f"✅ Renamed '{old_name}' to '{new_name}'."

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
        return f"✅ Removed {before - after} rows with null values. Now {after} rows."

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
            elif method in ["ffill", "bfill"]:
                if col:
                    self.df[col] = self.df[col].fillna(method=method)
                else:
                    self.df = self.df.fillna(method=method)
                return f"✅ Filled nulls using {method}."
        
        if col:
            count = self.df[col].isna().sum()
            self.df[col] = self.df[col].fillna(value)
            return f"✅ Filled {count} null values in '{col}' with '{value}'."
        else:
            self.df = self.df.fillna(value)
            return f"✅ Filled all null values with '{value}'."

    def _drop_duplicates(self, params: Dict) -> str:
        cols = params.get("columns")
        keep = params.get("keep", "first")
        before = len(self.df)
        if cols:
            self.df = self.df.drop_duplicates(subset=cols, keep=keep)
        else:
            self.df = self.df.drop_duplicates(keep=keep)
        after = len(self.df)
        return f"✅ Removed {before - after} duplicate rows. Now {after} rows."

    def _filter_rows(self, params: Dict) -> str:
        col = params.get("column")
        operator = params.get("operator", "==")
        value = params.get("value")
        
        if col not in self.df.columns:
            # Try case-insensitive match
            matches = [c for c in self.df.columns if c.lower() == col.lower()]
            if matches:
                col = matches[0]
            else:
                # Try partial match
                matches = [c for c in self.df.columns if col.lower() in c.lower() or c.lower() in col.lower()]
                if matches:
                    col = matches[0]
                else:
                    return f"Column '{col}' not found. Available: {list(self.df.columns)}"
        
        before = len(self.df)
        
        # Try to convert value to match column dtype
        try:
            col_dtype = self.df[col].dtype
            if pd.api.types.is_integer_dtype(col_dtype):
                # Try converting value to int
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    pass
            elif pd.api.types.is_float_dtype(col_dtype):
                # Try converting value to float
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    pass
        except Exception:
            pass  # Keep original value if conversion fails
        
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
        return f"✅ Filtered to {after} rows (removed {before - after})."

    def _drop_rows(self, params: Dict) -> str:
        col = params.get("column")
        operator = params.get("operator", "==")
        value = params.get("value")
        
        if col not in self.df.columns:
            # Try case-insensitive match
            matches = [c for c in self.df.columns if c.lower() == col.lower()]
            if matches:
                col = matches[0]
            else:
                # Try partial match
                matches = [c for c in self.df.columns if col.lower() in c.lower() or c.lower() in col.lower()]
                if matches:
                    col = matches[0]
                else:
                    return f"Column '{col}' not found. Available: {list(self.df.columns)}"
        
        before = len(self.df)
        
        # Try to convert value to match column dtype
        try:
            col_dtype = self.df[col].dtype
            if pd.api.types.is_integer_dtype(col_dtype):
                # Try converting value to int
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    pass
            elif pd.api.types.is_float_dtype(col_dtype):
                # Try converting value to float
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    pass
        except Exception:
            pass  # Keep original value if conversion fails
        
        if operator == "==":
            self.df = self.df[self.df[col] != value]
        elif operator == "contains":
            self.df = self.df[~self.df[col].astype(str).str.contains(str(value), case=False, na=False)]
        
        after = len(self.df)
        return f"✅ Dropped {before - after} rows. Now {after} rows."

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
            return f"✅ Converted '{col}' to {dtype}."
        except Exception as e:
            return f"❌ Could not convert '{col}' to {dtype}: {str(e)}"

    def _strip_whitespace(self, params: Dict) -> str:
        col = params.get("column")
        if col:
            if col not in self.df.columns:
                return f"Column '{col}' not found."
            if self.df[col].dtype == "object":
                self.df[col] = self.df[col].str.strip()
            return f"✅ Stripped whitespace from '{col}'."
        else:
            for c in self.df.select_dtypes(include=["object"]).columns:
                self.df[c] = self.df[c].str.strip()
            return "✅ Stripped whitespace from all text columns."

    def _lowercase(self, params: Dict) -> str:
        col = params.get("column")
        if col not in self.df.columns:
            return f"Column '{col}' not found."
        self.df[col] = self.df[col].astype(str).str.lower()
        return f"✅ Converted '{col}' to lowercase."

    def _uppercase(self, params: Dict) -> str:
        col = params.get("column")
        if col not in self.df.columns:
            return f"Column '{col}' not found."
        self.df[col] = self.df[col].astype(str).str.upper()
        return f"✅ Converted '{col}' to uppercase."

    def _replace_values(self, params: Dict) -> str:
        col = params.get("column")
        old_value = params.get("old_value")
        new_value = params.get("new_value")
        
        if col:
            if col not in self.df.columns:
                return f"Column '{col}' not found."
            count = (self.df[col] == old_value).sum()
            self.df[col] = self.df[col].replace(old_value, new_value)
            return f"✅ Replaced {count} occurrences of '{old_value}' with '{new_value}' in '{col}'."
        else:
            self.df = self.df.replace(old_value, new_value)
            return f"✅ Replaced all '{old_value}' with '{new_value}'."

    def _sort(self, params: Dict) -> str:
        col = params.get("column")
        ascending = params.get("ascending", True)
        
        if col not in self.df.columns:
            return f"Column '{col}' not found."
        
        self.df = self.df.sort_values(by=col, ascending=ascending)
        order = "ascending" if ascending else "descending"
        return f"✅ Sorted by '{col}' ({order})."

    def _keep_columns(self, params: Dict) -> str:
        cols = params.get("columns", [])
        valid_cols = [c for c in cols if c in self.df.columns]
        if not valid_cols:
            return f"None of the specified columns found."
        self.df = self.df[valid_cols]
        return f"✅ Kept {len(valid_cols)} columns: {valid_cols}"

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
            return f"✅ Added column '{name}' based on '{from_column}'."
        elif value is not None:
            self.df[name] = value
            return f"✅ Added column '{name}' with value '{value}'."
        else:
            self.df[name] = None
            return f"✅ Added empty column '{name}'."

    def _add_conditional_column(self, params: Dict) -> str:
        name = params.get("name")
        condition_col = params.get("condition_column")
        operator = params.get("operator", ">")
        threshold = params.get("threshold")
        true_value = params.get("true_value", True)
        false_value = params.get("false_value", False)
        conditions = params.get("conditions")  # List of conditions for ranges
        
        if not name or not condition_col:
            return "Column name and condition column are required."
        
        if condition_col not in self.df.columns:
            # Try case-insensitive match
            matches = [c for c in self.df.columns if c.lower() == condition_col.lower()]
            if matches:
                condition_col = matches[0]
            else:
                return f"Column '{condition_col}' not found."
        
        try:
            # Support multiple conditions/ranges
            # Handle cases where conditions might be a string representation
            if isinstance(conditions, str):
                try:
                    import json
                    conditions = json.loads(conditions)
                except:
                    pass
            
            if conditions and isinstance(conditions, list):
                # Multiple conditions/ranges (e.g., < 30, 30-60, > 60)
                # Build conditions and choices for np.select
                cond_list = []
                choice_list = []
                
                for cond in conditions:
                    op = cond.get("operator", ">")
                    val = cond.get("value")
                    result_val = cond.get("result")
                    
                    if op == "<":
                        cond_list.append(self.df[condition_col] < val)
                    elif op == "<=":
                        cond_list.append(self.df[condition_col] <= val)
                    elif op == ">":
                        cond_list.append(self.df[condition_col] > val)
                    elif op == ">=":
                        cond_list.append(self.df[condition_col] >= val)
                    elif op == "==":
                        cond_list.append(self.df[condition_col] == val)
                    elif op == "!=":
                        cond_list.append(self.df[condition_col] != val)
                    elif op == "between" or op == "range":

                        val1 = cond.get("value1") or cond.get("min")
                        val2 = cond.get("value2") or cond.get("max")
                        if val1 is not None and val2 is not None:
                            cond_list.append((self.df[condition_col] >= val1) & (self.df[condition_col] <= val2))
                        else:
                            continue
                    else:
                        continue
                    
                    choice_list.append(result_val)
                
                if cond_list and choice_list:
                    # Use np.select to apply conditions
                    # Order matters: more specific conditions should come first
                    # For ranges like <30, 30-60, >60, process in order: <30, then 30-60, then >60
                    default_val = params.get("default_value", "Unknown")
                    self.df[name] = np.select(cond_list, choice_list, default=default_val)
                    return f"✅ Added column '{name}' with {len(conditions)} conditional ranges."
                else:
                    return "❌ No valid conditions provided."
            
            # Single condition (backward compatibility)
            if operator == ">":
                self.df[name] = self.df[condition_col] > threshold
            elif operator == "<":
                self.df[name] = self.df[condition_col] < threshold
            elif operator == ">=":
                self.df[name] = self.df[condition_col] >= threshold
            elif operator == "<=":
                self.df[name] = self.df[condition_col] <= threshold
            elif operator == "==":
                self.df[name] = self.df[condition_col] == threshold
            elif operator == "!=":
                self.df[name] = self.df[condition_col] != threshold
            elif operator == "contains":
                self.df[name] = self.df[condition_col].astype(str).str.contains(str(threshold), case=False, na=False)
            
            # Apply custom true/false values if not boolean
            if true_value is not True or false_value is not False:
                self.df[name] = self.df[name].map({True: true_value, False: false_value})
            
            return f"✅ Added column '{name}' based on condition: {condition_col} {operator} {threshold}."
        except Exception as e:
            return f"❌ Error creating conditional column: {str(e)}"

    def get_result(self) -> pd.DataFrame:
        return self.df

    def get_summary(self) -> str:
        """Get a summary of changes made."""
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


def parse_intent(message: str, columns: List[str]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Parse user message to extract operation intent."""
    msg = message.lower().strip()
    
    # Drop column patterns
    drop_col_patterns = [
        r"(?:remove|drop|delete|get rid of)\s+(?:the\s+)?(?:column\s+)?['\"]?(\w+)['\"]?\s*(?:column)?",
        r"(?:remove|drop|delete)\s+['\"]?(\w+)['\"]?\s+column",
    ]
    for pattern in drop_col_patterns:
        match = re.search(pattern, msg)
        if match:
            col = match.group(1)
            # Try to match against actual columns
            actual_col = _find_column(col, columns)
            if actual_col:
                return ("drop_column", {"column": actual_col})
    
    # Rename column
    rename_patterns = [
        r"rename\s+['\"]?(\w+)['\"]?\s+(?:to|as)\s+['\"]?(\w+)['\"]?",
        r"change\s+['\"]?(\w+)['\"]?\s+(?:to|name)\s+['\"]?(\w+)['\"]?",
    ]
    for pattern in rename_patterns:
        match = re.search(pattern, msg)
        if match:
            old = _find_column(match.group(1), columns) or match.group(1)
            new = match.group(2)
            return ("rename_column", {"old_name": old, "new_name": new})
    
    # Drop nulls
    if any(x in msg for x in ["remove null", "drop null", "delete null", "remove missing", "drop missing", "remove empty", "drop empty", "clean null"]):
        col_match = re.search(r"(?:in|from)\s+['\"]?(\w+)['\"]?", msg)
        if col_match:
            col = _find_column(col_match.group(1), columns)
            return ("drop_nulls", {"column": col})
        return ("drop_nulls", {})
    
    # Fill nulls
    fill_patterns = [
        r"fill\s+(?:null|missing|empty)s?\s+(?:in\s+)?['\"]?(\w+)['\"]?\s+with\s+['\"]?(.+?)['\"]?$",
        r"replace\s+(?:null|missing)s?\s+(?:in\s+)?['\"]?(\w+)['\"]?\s+with\s+['\"]?(.+?)['\"]?$",
    ]
    for pattern in fill_patterns:
        match = re.search(pattern, msg)
        if match:
            col = _find_column(match.group(1), columns) or match.group(1)
            value = match.group(2).strip("'\"")
            # Check if it's a method
            if value in ["mean", "median", "mode", "average"]:
                return ("fill_nulls", {"column": col, "method": value if value != "average" else "mean"})
            return ("fill_nulls", {"column": col, "value": value})
    
    # Drop duplicates
    if any(x in msg for x in ["remove duplicate", "drop duplicate", "delete duplicate", "dedupe", "deduplicate"]):
        return ("drop_duplicates", {})
    
    # Filter/keep rows
    filter_patterns = [
        r"(?:keep|filter)\s+(?:only\s+)?(?:rows\s+)?where\s+['\"]?(\w+)['\"]?\s*(==|!=|>|<|>=|<=|contains|is|equals?)\s*['\"]?(.+?)['\"]?$",
        r"(?:show|get)\s+(?:only\s+)?(?:rows\s+)?where\s+['\"]?(\w+)['\"]?\s*(==|!=|>|<|>=|<=|contains|is|equals?)\s*['\"]?(.+?)['\"]?$",
    ]
    for pattern in filter_patterns:
        match = re.search(pattern, msg)
        if match:
            col = _find_column(match.group(1), columns) or match.group(1)
            op = match.group(2)
            if op in ["is", "equals", "equal"]:
                op = "=="
            value = match.group(3).strip("'\"")
            try:
                value = float(value) if "." in value else int(value)
            except ValueError:
                pass
            return ("filter_rows", {"column": col, "operator": op, "value": value})
    
    # Drop rows
    drop_row_patterns = [
        r"(?:remove|drop|delete)\s+(?:rows?\s+)?where\s+['\"]?(\w+)['\"]?\s*(==|!=|>|<|>=|<=|contains|is|equals?)\s*['\"]?(.+?)['\"]?$",
    ]
    for pattern in drop_row_patterns:
        match = re.search(pattern, msg)
        if match:
            col = _find_column(match.group(1), columns) or match.group(1)
            op = match.group(2)
            if op in ["is", "equals", "equal"]:
                op = "=="
            value = match.group(3).strip("'\"")
            try:
                value = float(value) if "." in value else int(value)
            except ValueError:
                pass
            return ("drop_rows", {"column": col, "operator": op, "value": value})
    
    # Convert type
    type_patterns = [
        r"convert\s+['\"]?(\w+)['\"]?\s+to\s+(int|integer|float|number|string|text|date|datetime|bool|boolean)",
        r"(?:change|make)\s+['\"]?(\w+)['\"]?\s+(?:to\s+)?(?:a\s+)?(int|integer|float|number|string|text|date|datetime|bool|boolean)",
    ]
    for pattern in type_patterns:
        match = re.search(pattern, msg)
        if match:
            col = _find_column(match.group(1), columns) or match.group(1)
            dtype = match.group(2)
            return ("convert_type", {"column": col, "dtype": dtype})
    
    # Strip whitespace
    if any(x in msg for x in ["strip whitespace", "trim whitespace", "remove whitespace", "clean whitespace", "strip spaces", "trim spaces"]):
        col_match = re.search(r"(?:in|from)\s+['\"]?(\w+)['\"]?", msg)
        if col_match:
            col = _find_column(col_match.group(1), columns)
            return ("strip_whitespace", {"column": col})
        return ("strip_whitespace", {})
    
    # Lowercase
    if any(x in msg for x in ["lowercase", "lower case", "to lower"]):
        col_match = re.search(r"['\"]?(\w+)['\"]?\s+(?:to\s+)?lowercase|lowercase\s+['\"]?(\w+)['\"]?", msg)
        if col_match:
            col = col_match.group(1) or col_match.group(2)
            col = _find_column(col, columns) or col
            return ("lowercase", {"column": col})
    
    # Uppercase
    if any(x in msg for x in ["uppercase", "upper case", "to upper"]):
        col_match = re.search(r"['\"]?(\w+)['\"]?\s+(?:to\s+)?uppercase|uppercase\s+['\"]?(\w+)['\"]?", msg)
        if col_match:
            col = col_match.group(1) or col_match.group(2)
            col = _find_column(col, columns) or col
            return ("uppercase", {"column": col})
    
    # Sort
    sort_patterns = [
        r"sort\s+(?:by\s+)?['\"]?(\w+)['\"]?\s*(asc|desc|ascending|descending)?",
        r"order\s+(?:by\s+)?['\"]?(\w+)['\"]?\s*(asc|desc|ascending|descending)?",
    ]
    for pattern in sort_patterns:
        match = re.search(pattern, msg)
        if match:
            col = _find_column(match.group(1), columns) or match.group(1)
            order = match.group(2) if match.group(2) else "asc"
            ascending = order.lower().startswith("asc")
            return ("sort", {"column": col, "ascending": ascending})
    
    # Replace values
    replace_patterns = [
        r"replace\s+['\"]?(.+?)['\"]?\s+with\s+['\"]?(.+?)['\"]?\s+(?:in\s+)?['\"]?(\w+)['\"]?",
        r"(?:in\s+)?['\"]?(\w+)['\"]?\s+replace\s+['\"]?(.+?)['\"]?\s+with\s+['\"]?(.+?)['\"]?",
    ]
    for pattern in replace_patterns:
        match = re.search(pattern, msg)
        if match:
            if "in" in msg and pattern.startswith(r"replace"):
                old_val, new_val, col = match.groups()
            else:
                col, old_val, new_val = match.groups()
            col = _find_column(col, columns) or col
            return ("replace_values", {"column": col, "old_value": old_val, "new_value": new_val})
    
    # Add conditional column (e.g., "add column is_senior where age > 45")
    add_cond_patterns = [
        r"(?:add|create|make)\s+(?:a\s+)?(?:new\s+)?column\s+(?:called\s+)?['\"]?(\w+)['\"]?\s+(?:where|if|when|based on)\s+['\"]?(\w+)['\"]?\s*(>|<|>=|<=|==|!=|is|equals?)\s*['\"]?(.+?)['\"]?$",
        r"(?:add|create)\s+['\"]?(\w+)['\"]?\s+column\s+(?:where|if|when)\s+['\"]?(\w+)['\"]?\s*(>|<|>=|<=|==|!=|is|equals?)\s*['\"]?(.+?)['\"]?$",
        r"(?:add|create)\s+(?:a\s+)?column\s+(?:to\s+)?(?:show|indicate|mark)\s+(?:if|whether|when)\s+['\"]?(\w+)['\"]?\s*(>|<|>=|<=|==|!=|is)\s*['\"]?(.+?)['\"]?",
    ]
    for pattern in add_cond_patterns:
        match = re.search(pattern, msg)
        if match:
            groups = match.groups()
            if len(groups) == 4:
                name, cond_col, op, threshold = groups
            else:
                # Infer column name from condition
                cond_col, op, threshold = groups
                name = f"{cond_col}_flag"
            
            cond_col = _find_column(cond_col, columns) or cond_col
            if op in ["is", "equals", "equal"]:
                op = "=="
            try:
                threshold = float(threshold) if "." in str(threshold) else int(threshold)
            except ValueError:
                pass
            return ("add_conditional_column", {
                "name": name,
                "condition_column": cond_col,
                "operator": op,
                "threshold": threshold,
            })
    
    # Simple add column (e.g., "add column status with value active")
    add_col_patterns = [
        r"(?:add|create|make)\s+(?:a\s+)?(?:new\s+)?column\s+(?:called\s+)?['\"]?(\w+)['\"]?\s+(?:with\s+(?:value\s+)?)?['\"]?(.+?)['\"]?$",
        r"(?:add|create)\s+['\"]?(\w+)['\"]?\s+column",
    ]
    for pattern in add_col_patterns:
        match = re.search(pattern, msg)
        if match:
            groups = match.groups()
            name = groups[0]
            value = groups[1] if len(groups) > 1 else None
            # Don't match if this looks like a conditional
            if value and any(x in str(value).lower() for x in ["where", "if", "when", ">", "<", "="]):
                continue
            return ("add_column", {"name": name, "value": value})
    
    return None


def _find_column(name: str, columns: List[str]) -> Optional[str]:
    """Find a column by name (case-insensitive, partial match)."""
    name_lower = name.lower()
    # Exact match (case-insensitive)
    for col in columns:
        if col.lower() == name_lower:
            return col
    # Partial match
    for col in columns:
        if name_lower in col.lower() or col.lower() in name_lower:
            return col
    return None
