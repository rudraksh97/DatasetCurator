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


class GetDistinctValuesHandler(BaseQueryHandler):
    """Handler for getting distinct values from a column."""
    
    @property
    def query_type(self) -> str:
        return "distinct_values"
    
    @property
    def supported_functions(self) -> List[str]:
        return ["get_distinct_values"]
    
    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> QueryResult:
        column = arguments.get("column")
        limit = arguments.get("limit", 50)
        
        if not column:
            return self._error("Missing column parameter")
        
        col_match = find_column_case_insensitive(df, column)
        if not col_match:
            return self._error(f"Column '{column}' not found. Available: {', '.join(df.columns[:10])}")
        
        try:
            distinct_values = df[col_match].dropna().unique()
            total_distinct = len(distinct_values)
            
            # Sort values if possible
            try:
                distinct_values = sorted(distinct_values)
            except TypeError:
                # Can't sort mixed types, keep as is
                pass
            
            # Apply limit
            limited_values = distinct_values[:limit] if limit else distinct_values
            limited_values = [convert_to_native_type(v) for v in limited_values]
            
            return self._success(
                column=col_match,
                distinct_values=limited_values,
                total_distinct=total_distinct,
                showing=len(limited_values),
                has_more=total_distinct > len(limited_values),
            )
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


class AuditDataQualityHandler(BaseQueryHandler):
    """Handler for comprehensive data quality auditing."""
    
    @property
    def query_type(self) -> str:
        return "audit"
    
    @property
    def supported_functions(self) -> List[str]:
        return ["audit_data_quality"]
    
    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> QueryResult:
        include_samples = arguments.get("include_sample_issues", True)
        
        issues = []
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "issues_found": 0,
        }
        
        # 1. Check for null values
        null_counts = df.isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]
        if len(columns_with_nulls) > 0:
            null_info = {
                "issue_type": "null_values",
                "severity": "warning",
                "affected_columns": len(columns_with_nulls),
                "details": {
                    col: {"null_count": int(count), "null_percentage": round(count / len(df) * 100, 2)}
                    for col, count in columns_with_nulls.items()
                }
            }
            issues.append(null_info)
            summary["issues_found"] += len(columns_with_nulls)
        
        # 2. Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            dup_info = {
                "issue_type": "duplicate_rows",
                "severity": "warning",
                "count": int(duplicate_count),
                "percentage": round(duplicate_count / len(df) * 100, 2),
            }
            if include_samples:
                # Get first duplicate
                dup_mask = df.duplicated(keep=False)
                sample_dup = df[dup_mask].head(1).to_dict(orient="records")
                if sample_dup:
                    dup_info["sample"] = {k: convert_to_native_type(v) for k, v in sample_dup[0].items()}
            issues.append(dup_info)
            summary["issues_found"] += 1
        
        # 3. Check for suspicious string values
        suspicious_patterns = ["unknown", "n/a", "na", "null", "none", "undefined", "-", "?", ""]
        suspicious_findings = {}
        
        for col in df.select_dtypes(include=["object"]).columns:
            col_lower = df[col].astype(str).str.lower().str.strip()
            suspicious_mask = col_lower.isin(suspicious_patterns)
            suspicious_count = suspicious_mask.sum()
            
            if suspicious_count > 0:
                suspicious_values = df[suspicious_mask][col].unique()[:5]
                suspicious_findings[col] = {
                    "count": int(suspicious_count),
                    "percentage": round(suspicious_count / len(df) * 100, 2),
                    "sample_values": [convert_to_native_type(v) for v in suspicious_values],
                }
        
        if suspicious_findings:
            issues.append({
                "issue_type": "suspicious_values",
                "severity": "info",
                "description": "Values like 'Unknown', 'N/A', 'null' found",
                "affected_columns": len(suspicious_findings),
                "details": suspicious_findings,
            })
            summary["issues_found"] += len(suspicious_findings)
        
        # 4. Check for potential outliers in numeric columns (using IQR)
        outlier_findings = {}
        for col in df.select_dtypes(include=["number"]).columns:
            col_data = df[col].dropna()
            if len(col_data) < 10:
                continue
            
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            if len(outliers) > 0:
                outlier_findings[col] = {
                    "count": int(len(outliers)),
                    "percentage": round(len(outliers) / len(col_data) * 100, 2),
                    "bounds": {
                        "lower": convert_to_native_type(lower_bound),
                        "upper": convert_to_native_type(upper_bound),
                    },
                    "min_value": convert_to_native_type(col_data.min()),
                    "max_value": convert_to_native_type(col_data.max()),
                }
                if include_samples:
                    outlier_findings[col]["sample_outliers"] = [
                        convert_to_native_type(v) for v in outliers.head(3).tolist()
                    ]
        
        if outlier_findings:
            issues.append({
                "issue_type": "potential_outliers",
                "severity": "info",
                "description": "Values outside 1.5*IQR range",
                "affected_columns": len(outlier_findings),
                "details": outlier_findings,
            })
            summary["issues_found"] += len(outlier_findings)
        
        # 5. Check for columns with very low cardinality (potential encoding issues)
        low_cardinality = {}
        for col in df.columns:
            unique_count = df[col].nunique()
            if unique_count == 1:
                low_cardinality[col] = {
                    "unique_values": 1,
                    "note": "Column has only one unique value - might be useless",
                    "value": convert_to_native_type(df[col].iloc[0]) if len(df) > 0 else None,
                }
            elif unique_count == 2 and df[col].dtype == "object":
                values = df[col].unique().tolist()
                low_cardinality[col] = {
                    "unique_values": 2,
                    "note": "Binary column - consider converting to boolean",
                    "values": [convert_to_native_type(v) for v in values],
                }
        
        if low_cardinality:
            issues.append({
                "issue_type": "low_cardinality",
                "severity": "info",
                "affected_columns": len(low_cardinality),
                "details": low_cardinality,
            })
        
        # 6. Check for mixed types in object columns
        mixed_type_cols = {}
        for col in df.select_dtypes(include=["object"]).columns:
            # Check if column has numeric-like strings mixed with text
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            
            numeric_count = col_data.apply(lambda x: str(x).replace(".", "").replace("-", "").isdigit()).sum()
            if 0 < numeric_count < len(col_data) * 0.9 and numeric_count > len(col_data) * 0.1:
                mixed_type_cols[col] = {
                    "numeric_values": int(numeric_count),
                    "non_numeric_values": int(len(col_data) - numeric_count),
                    "note": "Column has mixed numeric and text values",
                }
        
        if mixed_type_cols:
            issues.append({
                "issue_type": "mixed_types",
                "severity": "warning",
                "affected_columns": len(mixed_type_cols),
                "details": mixed_type_cols,
            })
            summary["issues_found"] += len(mixed_type_cols)
        
        # Build quality score
        if len(df) > 0:
            null_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))
            dup_rate = duplicate_count / len(df)
            quality_score = max(0, round((1 - null_rate - dup_rate) * 100, 1))
        else:
            quality_score = 0
        
        summary["quality_score"] = quality_score
        summary["quality_rating"] = (
            "Excellent" if quality_score >= 95 else
            "Good" if quality_score >= 85 else
            "Fair" if quality_score >= 70 else
            "Needs Attention"
        )
        
        return self._success(
            summary=summary,
            issues=issues,
            recommendations=self._get_recommendations(issues),
        )
    
    def _get_recommendations(self, issues: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on issues found."""
        recommendations = []
        
        for issue in issues:
            issue_type = issue.get("issue_type")
            
            if issue_type == "null_values":
                recommendations.append("Consider filling null values with mean/median or dropping rows with nulls")
            elif issue_type == "duplicate_rows":
                recommendations.append("Consider removing duplicate rows with 'drop duplicates'")
            elif issue_type == "suspicious_values":
                recommendations.append("Review suspicious values like 'Unknown' or 'N/A' - they might need cleaning")
            elif issue_type == "potential_outliers":
                recommendations.append("Review potential outliers - they might be data entry errors or valid extremes")
            elif issue_type == "mixed_types":
                recommendations.append("Columns with mixed types may cause issues - consider converting to consistent type")
        
        if not recommendations:
            recommendations.append("No major issues found - dataset looks clean!")
        
        return recommendations
