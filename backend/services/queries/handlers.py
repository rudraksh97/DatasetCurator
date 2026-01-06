"""Concrete query handler implementations.

This module provides concrete implementations of query handlers
for different query types (column search, row search, statistics, etc.).
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from protocols import QueryResult
from services.queries.base import (
    BaseQueryHandler,
    apply_filter_to_df,
    coerce_filter_value,
    convert_to_native_type,
    find_column_case_insensitive,
)


class FindColumnsHandler(BaseQueryHandler):
    """Handler for finding columns by keyword matching."""
    
    @property
    def query_type(self) -> str:
        return "column_search"
    
    @property
    def supported_functions(self) -> List[str]:
        return ["find_columns"]
    
    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> QueryResult:
        keywords = arguments.get("keywords", [])
        if not keywords:
            return self._error("Missing keywords parameter")
        
        matched_columns = []
        keyword_lower = [k.lower() for k in keywords]
        
        for col in df.columns:
            col_lower = col.lower()
            for keyword in keyword_lower:
                if keyword in col_lower or col_lower in keyword:
                    matched_columns.append({"column": col, "matched_keyword": keyword})
                    break
        
        if not matched_columns:
            return self._error(
                f"No columns found matching keywords: {', '.join(keywords)}. "
                f"Available columns: {', '.join(df.columns.tolist())}"
            )
        
        return self._success(
            keywords=keywords,
            matched_columns=matched_columns,
            column_names=[c["column"] for c in matched_columns],
        )


class SearchRowsHandler(BaseQueryHandler):
    """Handler for searching rows by partial match."""
    
    @property
    def query_type(self) -> str:
        return "row_search"
    
    @property
    def supported_functions(self) -> List[str]:
        return ["search_rows"]
    
    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> QueryResult:
        column = arguments.get("column")
        keyword = arguments.get("keyword", "")
        limit = arguments.get("limit", 5)
        
        if not column:
            return self._error("Missing column parameter")
        
        col_match = find_column_case_insensitive(df, column)
        if not col_match:
            return self._error(f"Column '{column}' not found. Available: {', '.join(df.columns[:10])}")
        
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
            return self._error(f"No rows found containing '{keyword}' in column '{col_match}'")
        
        found_values = [convert_to_native_type(v) for v in matches[col_match].tolist()]
        
        return self._success(
            column=col_match,
            keyword=keyword,
            matches=found_values,
            total_matches=int(mask.sum()),
            showing=len(found_values),
        )


class GetRowHandler(BaseQueryHandler):
    """Handler for getting a row by exact match."""
    
    @property
    def query_type(self) -> str:
        return "row_lookup"
    
    @property
    def supported_functions(self) -> List[str]:
        return ["get_row"]
    
    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> QueryResult:
        column = arguments.get("column")
        value = arguments.get("value")
        
        if not column or value is None:
            return self._error("Missing column or value")
        
        value = coerce_filter_value(df, column, value)
        matches = df[df[column] == value]
        
        if len(matches) > 0:
            row_data = {k: convert_to_native_type(v) for k, v in matches.iloc[0].to_dict().items()}
            return self._success(data=row_data, row_count=int(len(matches)))
        
        return self._error(f"No rows found where {column} == {value}")


class GetValueHandler(BaseQueryHandler):
    """Handler for getting a specific value from a filtered row."""
    
    @property
    def query_type(self) -> str:
        return "value_lookup"
    
    @property
    def supported_functions(self) -> List[str]:
        return ["get_value"]
    
    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> QueryResult:
        column = arguments.get("column")
        filter_column = arguments.get("filter_column")
        filter_value = arguments.get("filter_value")
        
        if not all([column, filter_column, filter_value]):
            return self._error("Missing required parameters")
        
        filter_value = coerce_filter_value(df, filter_column, filter_value)
        matches = df[df[filter_column] == filter_value]
        
        if len(matches) > 0:
            value = convert_to_native_type(matches.iloc[0][column])
            return self._success(value=value, row_count=int(len(matches)))
        
        return self._error(f"No rows found where {filter_column} == {filter_value}")


class CalculateRatioHandler(BaseQueryHandler):
    """Handler for calculating ratio between columns."""
    
    @property
    def query_type(self) -> str:
        return "ratio_calculation"
    
    @property
    def supported_functions(self) -> List[str]:
        return ["calculate_ratio"]
    
    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> QueryResult:
        numerator_col = arguments.get("numerator_column")
        denominator_col = arguments.get("denominator_column")
        filter_column = arguments.get("filter_column")
        filter_value = arguments.get("filter_value")
        
        if not numerator_col or not denominator_col:
            return self._error("Missing numerator or denominator column")
        
        df_subset = df
        if filter_column and filter_value:
            filter_value = coerce_filter_value(df, filter_column, filter_value)
            matches = df[df[filter_column] == filter_value]
            if len(matches) == 0:
                return self._error(f"No rows found where {filter_column} == {filter_value}")
            df_subset = matches
        
        numerator = df_subset[numerator_col].sum()
        denominator = df_subset[denominator_col].sum()
        
        if denominator == 0:
            return self._error("Denominator is zero")
        
        return self._success(
            numerator=convert_to_native_type(numerator),
            denominator=convert_to_native_type(denominator),
            ratio=convert_to_native_type(numerator / denominator),
            row_count=int(len(df_subset)),
        )


class GetStatisticsHandler(BaseQueryHandler):
    """Handler for computing column statistics."""
    
    @property
    def query_type(self) -> str:
        return "statistics"
    
    @property
    def supported_functions(self) -> List[str]:
        return ["get_statistics"]
    
    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> QueryResult:
        column = arguments.get("column")
        filter_column = arguments.get("filter_column")
        filter_value = arguments.get("filter_value")
        
        if not column:
            return self._error("Missing column parameter")
        
        col_match = find_column_case_insensitive(df, column)
        if not col_match:
            return self._error(f"Column '{column}' not found. Available: {', '.join(df.columns[:10])}")
        
        df_subset = df
        filter_info = ""
        
        if filter_column and filter_value:
            df_subset, filter_info, error = apply_filter_to_df(df, filter_column, filter_value)
            if error:
                return self._error(error)
        
        if not pd.api.types.is_numeric_dtype(df_subset[col_match]):
            return self._error(f"Column '{col_match}' is not numeric. Cannot calculate mean/variance/std/min/max.")
        
        stats = {
            "count": convert_to_native_type(df_subset[col_match].count()),
            "nulls": convert_to_native_type(df_subset[col_match].isna().sum()),
            "mean": convert_to_native_type(df_subset[col_match].mean()),
            "variance": convert_to_native_type(df_subset[col_match].var()),
            "std": convert_to_native_type(df_subset[col_match].std()),
            "min": convert_to_native_type(df_subset[col_match].min()),
            "max": convert_to_native_type(df_subset[col_match].max()),
            "filter_info": filter_info if filter_info else None,
        }
        return self._success(statistics=stats)


class GroupByHandler(BaseQueryHandler):
    """Handler for group-by aggregation queries."""
    
    @property
    def query_type(self) -> str:
        return "groupby"
    
    @property
    def supported_functions(self) -> List[str]:
        return ["group_by"]
    
    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> QueryResult:
        column = arguments.get("column")
        agg_function = arguments.get("aggregation", "count")
        agg_column = arguments.get("aggregation_column")
        
        if not column:
            return self._error("Missing column parameter")
        
        col_match = find_column_case_insensitive(df, column)
        if not col_match:
            return self._error(f"Column '{column}' not found")
        
        try:
            grouped = df.groupby(col_match)
            
            if agg_function == "count":
                result = grouped.size().to_dict()
            elif agg_column:
                if agg_column not in df.columns:
                    return self._error(f"Aggregation column '{agg_column}' not found")
                agg_map = {"sum": "sum", "mean": "mean", "min": "min", "max": "max"}
                if agg_function in agg_map:
                    result = getattr(grouped[agg_column], agg_map[agg_function])().to_dict()
                else:
                    result = grouped.size().to_dict()
            else:
                result = grouped.size().to_dict()
            
            result = {str(k): convert_to_native_type(v) for k, v in result.items()}
            
            return self._success(
                group_by=col_match,
                aggregation=agg_function,
                results=result,
                total_groups=len(result),
            )
        except Exception as e:
            return self._error(str(e))


class GetRandomValueHandler(BaseQueryHandler):
    """Handler for getting random values or rows."""
    
    @property
    def query_type(self) -> str:
        return "random"
    
    @property
    def supported_functions(self) -> List[str]:
        return ["get_random_value"]
    
    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> QueryResult:
        column = arguments.get("column")
        
        if len(df) == 0:
            return self._error("Dataset is empty")
        
        try:
            if column:
                col_match = find_column_case_insensitive(df, column)
                if not col_match:
                    return self._error(f"Column '{column}' not found")
                
                non_null_values = df[col_match].dropna()
                if len(non_null_values) == 0:
                    return self._error(f"Column '{col_match}' has no non-null values")
                
                random_value = non_null_values.sample(n=1).iloc[0]
                return self._success(column=col_match, value=convert_to_native_type(random_value))
            else:
                random_row = df.sample(n=1).iloc[0]
                row_data = {k: convert_to_native_type(v) for k, v in random_row.to_dict().items()}
                return self._success(row=row_data)
        except Exception as e:
            return self._error(str(e))


class ListColumnsHandler(BaseQueryHandler):
    """Handler for listing all columns."""
    
    @property
    def query_type(self) -> str:
        return "metadata"
    
    @property
    def supported_functions(self) -> List[str]:
        return ["list_columns"]
    
    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> QueryResult:
        return self._success(columns=list(df.columns), row_count=int(len(df)))


class GetRowCountHandler(BaseQueryHandler):
    """Handler for getting row and column counts."""
    
    @property
    def query_type(self) -> str:
        return "metadata"
    
    @property
    def supported_functions(self) -> List[str]:
        return ["get_row_count"]
    
    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> QueryResult:
        return self._success(row_count=int(len(df)), column_count=int(len(df.columns)))
