"""Natural language data operations executor.

This module provides the DataOperator class for executing data transformations
on pandas DataFrames based on parsed operation intents from the LLM.

All operations map to canonical base primitives:

Row-Level (filter_rows, map_rows):
  - filter_rows, drop_rows, drop_nulls → filter_rows (boolean masking)
  - fill_nulls, strip_whitespace, lowercase, uppercase, replace_values → map_rows

Column/Schema (select_columns, rename_columns, add_column, drop_columns, cast_column_types):
  - keep_columns → select_columns
  - rename_column → rename_columns  
  - drop_column, drop_columns → drop_columns
  - add_column, add_conditional_column → add_column + map_rows
  - convert_type → cast_column_types

Dataset-Level (deduplicate_rows, sort_rows, limit_rows, sample_rows):
  - drop_duplicates → deduplicate_rows
  - sort → sort_rows
  - limit_rows → limit_rows (head/tail)
  - sample_rows → sample_rows (random sampling)

Grouping & Aggregation (group_rows, aggregate_groups):
  - group_aggregate → group_rows + aggregate_groups

Validation & Quality (validate_schema, quarantine_rows):
  - validate_schema → check columns, types, nulls, uniqueness, ranges
  - quarantine_rows → separate invalid rows (nulls, duplicates, range, regex, values)
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
        self.quarantined: Optional[pd.DataFrame] = None  # Holds rows that failed validation
        self.chart_config: Optional[Dict] = None  # Holds chart configuration if generated

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

    def _apply_string_transform(self, col: str, transform: str) -> str:
        """Base primitive: map_rows for string transformations."""
        if col not in self.df.columns:
            return f"Column '{col}' not found."
        
        transforms = {
            "lower": lambda s: s.astype(str).str.lower(),
            "upper": lambda s: s.astype(str).str.upper(),
            "strip": lambda s: s.str.strip() if s.dtype == "object" else s,
        }
        
        if transform in transforms:
            self.df[col] = transforms[transform](self.df[col])
            return f"Applied {transform} to '{col}'."
        return f"Unknown transform: {transform}"

    def execute(self, operation: str, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute a single operation and return success status and message."""
        op_map = {
            # Row filtering (filter_rows primitive)
            "filter_rows": self._filter_rows,
            "drop_rows": self._drop_rows,
            "drop_nulls": self._drop_nulls,
            # Row transforms (map_rows primitive)
            "fill_nulls": self._fill_nulls,
            "strip_whitespace": self._strip_whitespace,
            "lowercase": self._lowercase,
            "uppercase": self._uppercase,
            "replace_values": self._replace_values,
            # Column operations
            "drop_column": self._drop_column,
            "drop_columns": self._drop_columns,
            "keep_columns": self._keep_columns,
            "rename_column": self._rename_column,
            "add_column": self._add_column,
            "add_conditional_column": self._add_conditional_column,
            "convert_type": self._convert_type,
            # Dataset operations
            "drop_duplicates": self._drop_duplicates,
            "sort": self._sort,
            "limit_rows": self._limit_rows,
            "sample_rows": self._sample_rows,
            # Grouping & aggregation
            "group_aggregate": self._group_aggregate,
            # Validation & quality
            "validate_schema": self._validate_schema,
            "quarantine_rows": self._quarantine_rows,
            # Analysis operations
            "detect_outliers": self._detect_outliers,
            "get_statistics": self._get_statistics,
            "create_chart": self._create_chart,
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
        """Alias for drop_columns with single column. Base: drop_columns."""
        col = params.get("column")
        col = self._match_column(col) if col else None
        if not col:
            return f"Column not found. Available: {list(self.df.columns)}"
        return self._drop_columns({"columns": [col]})

    def _drop_columns(self, params: Dict) -> str:
        """Base primitive: drop_columns."""
        cols = params.get("columns", [])
        existing = [c for c in cols if c in self.df.columns]
        if existing:
            self.df = self.df.drop(columns=existing)
        return f"Dropped {len(existing)} columns: {existing}"

    def _rename_column(self, params: Dict) -> str:
        """Base primitive: rename_columns."""
        old_name = params.get("old_name")
        new_name = params.get("new_name")
        if old_name not in self.df.columns:
            return f"Column '{old_name}' not found."
        self.df = self.df.rename(columns={old_name: new_name})
        return f"Renamed '{old_name}' to '{new_name}'."

    def _drop_nulls(self, params: Dict) -> str:
        """Base primitive: filter_rows (NOT NULL predicate)."""
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
        """Base primitive: map_rows (fill transform)."""
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
        """Base primitive: deduplicate_rows."""
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
        """Base primitive: filter_rows (boolean masking)."""
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
        elif operator in ("between", "range"):
            # Support between operator: value is min, value2 is max
            value2 = params.get("value2")
            if value2 is not None:
                value2 = self._coerce_value(col, value2)
                self.df = self.df[(self.df[col] >= value) & (self.df[col] <= value2)]
        
        after = len(self.df)
        return f"Filtered to {after} rows (removed {before - after})."

    def _drop_rows(self, params: Dict) -> str:
        """Base primitive: filter_rows (inverted predicate)."""
        col = params.get("column")
        operator = params.get("operator", "==")
        value = params.get("value")

        col = self._match_column(col) if col else None
        if not col:
            return f"Column not found. Available: {list(self.df.columns)}"

        before = len(self.df)
        
        # Handle null value specially
        if value is None or (isinstance(value, str) and value.lower() in ("null", "none", "nan")):
            self.df = self.df[self.df[col].notna()]
        elif operator == "==":
            value = self._coerce_value(col, value)
            self.df = self.df[self.df[col] != value]
        elif operator == "!=":
            value = self._coerce_value(col, value)
            self.df = self.df[self.df[col] == value]
        elif operator == ">":
            value = self._coerce_value(col, value)
            self.df = self.df[self.df[col] <= value]
        elif operator == "<":
            value = self._coerce_value(col, value)
            self.df = self.df[self.df[col] >= value]
        elif operator == ">=":
            value = self._coerce_value(col, value)
            self.df = self.df[self.df[col] < value]
        elif operator == "<=":
            value = self._coerce_value(col, value)
            self.df = self.df[self.df[col] > value]
        elif operator == "contains":
            self.df = self.df[~self.df[col].astype(str).str.contains(str(value), case=False, na=False)]
        
        after = len(self.df)
        return f"Dropped {before - after} rows. Now {after} rows."

    def _convert_type(self, params: Dict) -> str:
        """Base primitive: cast_column_types."""
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
        """Base primitive: map_rows (strip transform)."""
        col = params.get("column")
        if col:
            return self._apply_string_transform(col, "strip")
        # Apply to all text columns
        for c in self.df.select_dtypes(include=["object"]).columns:
            self._apply_string_transform(c, "strip")
        return "Stripped whitespace from all text columns."

    def _lowercase(self, params: Dict) -> str:
        """Base primitive: map_rows (lower transform)."""
        return self._apply_string_transform(params.get("column"), "lower")

    def _uppercase(self, params: Dict) -> str:
        """Base primitive: map_rows (upper transform)."""
        return self._apply_string_transform(params.get("column"), "upper")

    def _replace_values(self, params: Dict) -> str:
        """Base primitive: map_rows (replace transform)."""
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
        """Base primitive: sort_rows."""
        col = params.get("column")
        ascending = params.get("ascending", True)
        
        if col not in self.df.columns:
            return f"Column '{col}' not found."
        
        self.df = self.df.sort_values(by=col, ascending=ascending)
        order = "ascending" if ascending else "descending"
        return f"Sorted by '{col}' ({order})."

    def _limit_rows(self, params: Dict) -> str:
        """Base primitive: limit_rows."""
        n = params.get("n", 100)
        from_end = params.get("from_end", False)
        
        before = len(self.df)
        if from_end:
            self.df = self.df.tail(n)
        else:
            self.df = self.df.head(n)
        
        position = "last" if from_end else "first"
        return f"Limited to {position} {len(self.df)} rows (was {before})."

    def _sample_rows(self, params: Dict) -> str:
        """Base primitive: sample_rows."""
        n = params.get("n")
        fraction = params.get("fraction")
        random_state = params.get("random_state", 42)
        
        before = len(self.df)
        
        if fraction:
            self.df = self.df.sample(frac=fraction, random_state=random_state)
        elif n:
            n = min(n, len(self.df))
            self.df = self.df.sample(n=n, random_state=random_state)
        else:
            return "Either 'n' or 'fraction' is required."
        
        return f"Sampled {len(self.df)} rows from {before}."

    def _group_aggregate(self, params: Dict) -> str:
        """Base primitive: group_rows + aggregate_groups."""
        group_by = params.get("group_by")
        aggregations = params.get("aggregations", {})
        
        if not group_by:
            return "group_by column(s) required."
        
        # Normalize group_by to list
        if isinstance(group_by, str):
            group_by = [group_by]
        
        # Validate columns exist
        missing = [c for c in group_by if c not in self.df.columns]
        if missing:
            return f"Columns not found: {missing}"
        
        if not aggregations:
            # Default: count
            self.df = self.df.groupby(group_by).size().reset_index(name="count")
            return f"Grouped by {group_by} with count. Now {len(self.df)} rows."
        
        # Build aggregation dict: {"col": "sum"} or {"col": ["sum", "mean"]}
        agg_dict = {}
        for col, agg in aggregations.items():
            if col not in self.df.columns:
                continue
            agg_dict[col] = agg
        
        if not agg_dict:
            return "No valid aggregation columns found."
        
        self.df = self.df.groupby(group_by).agg(agg_dict).reset_index()
        # Flatten multi-level columns if needed
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = ["_".join(col).strip("_") for col in self.df.columns]
        
        return f"Grouped by {group_by} with aggregations. Now {len(self.df)} rows."

    def _validate_schema(self, params: Dict) -> str:
        """Base primitive: validate_schema - check data against expected schema."""
        expected_columns = params.get("columns")  # List of required column names
        expected_types = params.get("types", {})  # {"col": "int|float|str|datetime|bool"}
        not_null = params.get("not_null", [])  # Columns that must not have nulls
        unique = params.get("unique", [])  # Columns that must have unique values
        value_ranges = params.get("ranges", {})  # {"col": {"min": 0, "max": 100}}
        
        issues = []
        
        # Check required columns exist
        if expected_columns:
            missing = [c for c in expected_columns if c not in self.df.columns]
            if missing:
                issues.append(f"Missing columns: {missing}")
        
        # Check column types
        type_map = {
            "int": ["int64", "int32", "Int64"],
            "float": ["float64", "float32", "Float64"],
            "str": ["object", "string"],
            "datetime": ["datetime64[ns]"],
            "bool": ["bool", "boolean"],
        }
        for col, expected_type in expected_types.items():
            if col not in self.df.columns:
                continue
            actual = str(self.df[col].dtype)
            valid_types = type_map.get(expected_type, [expected_type])
            if actual not in valid_types:
                issues.append(f"Column '{col}': expected {expected_type}, got {actual}")
        
        # Check not-null constraints
        for col in not_null:
            if col in self.df.columns:
                null_count = self.df[col].isna().sum()
                if null_count > 0:
                    issues.append(f"Column '{col}': {null_count} null values")
        
        # Check uniqueness
        for col in unique:
            if col in self.df.columns:
                dup_count = self.df[col].duplicated().sum()
                if dup_count > 0:
                    issues.append(f"Column '{col}': {dup_count} duplicate values")
        
        # Check value ranges
        for col, range_spec in value_ranges.items():
            if col not in self.df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            min_val = range_spec.get("min")
            max_val = range_spec.get("max")
            if min_val is not None:
                below = (self.df[col] < min_val).sum()
                if below > 0:
                    issues.append(f"Column '{col}': {below} values below {min_val}")
            if max_val is not None:
                above = (self.df[col] > max_val).sum()
                if above > 0:
                    issues.append(f"Column '{col}': {above} values above {max_val}")
        
        if issues:
            return f"Validation FAILED:\n- " + "\n- ".join(issues)
        return "Validation PASSED: Schema is valid."

    def _quarantine_rows(self, params: Dict) -> str:
        """Base primitive: quarantine_rows - separate invalid rows instead of dropping."""
        column = params.get("column")
        condition = params.get("condition", "null")  # null, duplicate, range, regex
        
        if not column or column not in self.df.columns:
            return f"Column '{column}' not found."
        
        # Build mask for rows to quarantine
        if condition == "null":
            mask = self.df[column].isna()
        elif condition == "duplicate":
            mask = self.df[column].duplicated(keep=params.get("keep", "first"))
        elif condition == "range":
            min_val = params.get("min")
            max_val = params.get("max")
            mask = pd.Series([False] * len(self.df), index=self.df.index)
            if min_val is not None:
                mask = mask | (self.df[column] < min_val)
            if max_val is not None:
                mask = mask | (self.df[column] > max_val)
        elif condition == "regex":
            pattern = params.get("pattern", "")
            # Quarantine rows that DON'T match the pattern (invalid format)
            mask = ~self.df[column].astype(str).str.match(pattern, na=False)
        elif condition == "values":
            # Quarantine rows with specific invalid values
            invalid_values = params.get("values", [])
            mask = self.df[column].isin(invalid_values)
        else:
            return f"Unknown condition: {condition}"
        
        quarantine_count = mask.sum()
        
        if quarantine_count == 0:
            return f"No rows to quarantine for condition '{condition}' on '{column}'."
        
        # Store quarantined rows
        quarantined = self.df[mask].copy()
        quarantined["_quarantine_reason"] = f"{condition} on {column}"
        
        if self.quarantined is None:
            self.quarantined = quarantined
        else:
            self.quarantined = pd.concat([self.quarantined, quarantined], ignore_index=True)
        
        # Remove from main dataframe
        self.df = self.df[~mask]
        
        return f"Quarantined {quarantine_count} rows ({condition} on '{column}'). Remaining: {len(self.df)} rows."

    def get_quarantined(self) -> Optional[pd.DataFrame]:
        """Get the quarantined rows DataFrame."""
        return self.quarantined

    def _keep_columns(self, params: Dict) -> str:
        """Base primitive: select_columns."""
        cols = params.get("columns", [])
        valid_cols = [c for c in cols if c in self.df.columns]
        if not valid_cols:
            return f"None of the specified columns found."
        self.df = self.df[valid_cols]
        return f"Kept {len(valid_cols)} columns: {valid_cols}"

    def _add_column(self, params: Dict) -> str:
        """Base primitive: add_column."""
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
        """Composite: add_column + map_rows (conditional transform)."""
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

    def _detect_outliers(self, params: Dict) -> str:
        """Detect and filter to show only outlier rows.
        
        Supports multiple methods:
        - IQR (default): Values outside Q1 - 1.5*IQR to Q3 + 1.5*IQR
        - z-score: Values more than N standard deviations from mean
        - custom: User-specified min/max bounds
        """
        column = params.get("column")
        method = params.get("method", "iqr").lower()
        threshold = params.get("threshold", 1.5)  # IQR multiplier or z-score threshold
        
        if not column:
            return "Column name is required."
        
        col = self._match_column(column)
        if not col:
            return f"Column '{column}' not found. Available: {list(self.df.columns)}"
        
        if not pd.api.types.is_numeric_dtype(self.df[col]):
            return f"Column '{col}' is not numeric. Outlier detection requires numeric data."
        
        col_data = self.df[col].dropna()
        if len(col_data) == 0:
            return f"Column '{col}' has no non-null values."
        
        # Calculate bounds based on method
        if method == "iqr":
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            method_desc = f"IQR method (threshold: {threshold}×IQR)"
        elif method in ("zscore", "z-score", "z_score"):
            mean = col_data.mean()
            std = col_data.std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            method_desc = f"Z-score method (threshold: {threshold}σ)"
        elif method == "custom":
            lower_bound = params.get("min")
            upper_bound = params.get("max")
            if lower_bound is None and upper_bound is None:
                return "Custom method requires 'min' and/or 'max' bounds."
            lower_bound = lower_bound if lower_bound is not None else float('-inf')
            upper_bound = upper_bound if upper_bound is not None else float('inf')
            method_desc = f"Custom bounds ({lower_bound} to {upper_bound})"
        else:
            return f"Unknown method: {method}. Use 'iqr', 'zscore', or 'custom'."
        
        # Filter to outliers only
        outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
        outlier_count = outlier_mask.sum()
        
        if outlier_count == 0:
            return f"No outliers found in '{col}' using {method_desc}. Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]"
        
        # Keep only outlier rows
        self.df = self.df[outlier_mask]
        
        return (
            f"Found {outlier_count} outliers in '{col}' using {method_desc}. "
            f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]. "
            f"Min outlier: {self.df[col].min():.2f}, Max outlier: {self.df[col].max():.2f}"
        )

    def _get_statistics(self, params: Dict) -> str:
        """Get detailed statistics for a column or the entire dataset."""
        column = params.get("column")
        
        if column:
            col = self._match_column(column)
            if not col:
                return f"Column '{column}' not found."
            
            if pd.api.types.is_numeric_dtype(self.df[col]):
                stats = self.df[col].describe()
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                
                result = f"**Statistics for '{col}':**\n"
                result += f"- Count: {int(stats['count'])}\n"
                result += f"- Mean: {stats['mean']:.2f}\n"
                result += f"- Std: {stats['std']:.2f}\n"
                result += f"- Min: {stats['min']:.2f}\n"
                result += f"- Q1 (25%): {q1:.2f}\n"
                result += f"- Median (50%): {stats['50%']:.2f}\n"
                result += f"- Q3 (75%): {q3:.2f}\n"
                result += f"- Max: {stats['max']:.2f}\n"
                result += f"- IQR: {iqr:.2f}\n"
                result += f"- Nulls: {self.df[col].isna().sum()}"
                return result
            else:
                value_counts = self.df[col].value_counts().head(10)
                result = f"**Statistics for '{col}' (categorical):**\n"
                result += f"- Unique values: {self.df[col].nunique()}\n"
                result += f"- Nulls: {self.df[col].isna().sum()}\n"
                result += f"- Top values:\n"
                for val, count in value_counts.items():
                    result += f"  - {val}: {count}\n"
                return result
        else:
            # Overall dataset statistics
            result = f"**Dataset Statistics:**\n"
            result += f"- Rows: {len(self.df)}\n"
            result += f"- Columns: {len(self.df.columns)}\n"
            result += f"- Memory: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n"
            result += f"- Nulls: {self.df.isna().sum().sum()} total\n"
            result += f"- Duplicates: {self.df.duplicated().sum()} rows\n"
            return result

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

    def _create_chart(self, params: Dict) -> str:
        """Create a chart visualization configuration."""
        chart_type = params.get("type", "bar").lower()
        
        # Map unknown types to closest supported type
        type_mapping = {
            "histogram": "bar",
            "box": "bar",
            "column": "bar",
            "plot": "line",
            "area": "line"
        }
        chart_type = type_mapping.get(chart_type, chart_type)
        
        x_col = params.get("x_column")
        y_col = params.get("y_column")
        title = params.get("title", f"{chart_type} chart")
        
        if not x_col:
            # Default to using index as x-axis
            x_col = "_RowID"
            # Add temporary column for plotting
            self.df[x_col] = range(1, len(self.df) + 1)
        else:
            x_col = self._match_column(x_col)
            if not x_col:
                return f"Column '{params.get('x_column')}' not found."
            
        if y_col:
            y_col = self._match_column(y_col)
            if not y_col:
                return f"Column '{params.get('y_column')}' not found."
        
        # Limit data for chart to avoid overwhelming the frontend
        # For categorical charts, we assume data is already aggregated if row count is small
        # otherwise we take top 50
        chart_data = self.df.head(50).replace({float('nan'): None})
        
        # Structure compatible with Chart.js / frontend
        self.chart_config = {
            "type": chart_type,
            "title": title,
            "data": chart_data.to_dict(orient="records"),
            "xField": x_col,
            "yField": y_col
        }
        
        return f"Created {chart_type} chart configuration: '{title}'."
