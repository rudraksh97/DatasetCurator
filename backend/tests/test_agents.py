"""Tests for data operations.

This module contains unit tests for the DataOperator class,
verifying that data transformations work correctly.
"""
import pandas as pd
import pytest
from agents.data_ops import DataOperator


def test_data_operator_drop_column():
    """Test dropping a column from the DataFrame."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    op = DataOperator(df)
    success, msg = op.execute("drop_column", {"column": "b"})
    assert success
    assert "b" not in op.get_result().columns


def test_data_operator_rename_column():
    """Test renaming a column."""
    df = pd.DataFrame({"old_name": [1, 2, 3]})
    op = DataOperator(df)
    success, msg = op.execute("rename_column", {"old_name": "old_name", "new_name": "new_name"})
    assert success
    assert "new_name" in op.get_result().columns
    assert "old_name" not in op.get_result().columns


def test_data_operator_filter_rows():
    """Test filtering rows based on a condition."""
    df = pd.DataFrame({"age": [10, 20, 30, 40]})
    op = DataOperator(df)
    success, msg = op.execute("filter_rows", {"column": "age", "operator": ">", "value": 25})
    assert success
    assert len(op.get_result()) == 2


def test_data_operator_add_conditional_column():
    """Test adding a column based on a condition."""
    df = pd.DataFrame({"age": [10, 20, 30]})
    op = DataOperator(df)
    success, msg = op.execute("add_conditional_column", {
        "name": "is_adult",
        "condition_column": "age",
        "operator": ">=",
        "threshold": 18,
        "true_value": "Yes",
        "false_value": "No"
    })
    assert success
    result = op.get_result()
    assert "is_adult" in result.columns
    assert list(result["is_adult"]) == ["No", "Yes", "Yes"]


def test_data_operator_invalid_column():
    """Test handling of non-existent column."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    op = DataOperator(df)
    success, msg = op.execute("drop_column", {"column": "nonexistent"})
    assert not success
    assert "not found" in msg.lower()


# =============================================================================
# Complex Multi-Step Transformation Tests
# =============================================================================

@pytest.fixture
def sample_dataset():
    """Create a realistic sample dataset for testing complex transformations."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "name": ["  Alice  ", "Bob", "Charlie", "  David", "Eve  ", "Frank", "Grace", "Heidi", "Ivan", "Judy"],
        "email": ["alice@test.com", "bob@test.com", "INVALID", "david@test.com", None, "frank@test.com", "grace@test.com", "heidi@test.com", "ivan@test.com", "judy@test.com"],
        "age": [25, 30, 35, None, 28, 150, 22, 45, -5, 33],
        "score": [85, 92, 78, 88, 95, 72, 88, None, 91, 85],
        "department": ["Engineering", "Sales", "Engineering", "HR", "Sales", "Engineering", "HR", "Sales", "Engineering", "HR"],
        "status": ["active", "ACTIVE", "Active", "inactive", "active", "ERROR", "active", "active", "active", "active"],
    })


class TestComplexTransformations:
    """Test complex multi-step transformation workflows."""

    def test_data_cleaning_pipeline(self, sample_dataset):
        """
        Test a realistic data cleaning pipeline:
        1. Strip whitespace from names
        2. Lowercase status for consistency
        3. Drop rows with null emails
        4. Filter out invalid ages (< 0 or > 120)
        5. Fill null scores with mean
        6. Drop duplicates
        7. Sort by name
        """
        op = DataOperator(sample_dataset)
        original_rows = len(sample_dataset)
        
        # Step 1: Strip whitespace from names
        success, msg = op.execute("strip_whitespace", {"column": "name"})
        assert success, f"Step 1 failed: {msg}"
        assert op.get_result()["name"].iloc[0] == "Alice"  # Was "  Alice  "
        
        # Step 2: Lowercase status
        success, msg = op.execute("lowercase", {"column": "status"})
        assert success, f"Step 2 failed: {msg}"
        assert all(s == s.lower() for s in op.get_result()["status"])
        
        # Step 3: Drop rows with null emails
        success, msg = op.execute("drop_nulls", {"column": "email"})
        assert success, f"Step 3 failed: {msg}"
        assert op.get_result()["email"].isna().sum() == 0
        
        # Step 4: Filter out invalid ages (keep 0 <= age <= 120)
        success, msg = op.execute("filter_rows", {"column": "age", "operator": ">=", "value": 0})
        assert success, f"Step 4a failed: {msg}"
        success, msg = op.execute("filter_rows", {"column": "age", "operator": "<=", "value": 120})
        assert success, f"Step 4b failed: {msg}"
        
        # Step 5: Fill null scores with mean
        success, msg = op.execute("fill_nulls", {"column": "score", "method": "mean"})
        assert success, f"Step 5 failed: {msg}"
        assert op.get_result()["score"].isna().sum() == 0
        
        # Step 6: Drop duplicates
        success, msg = op.execute("drop_duplicates", {})
        assert success, f"Step 6 failed: {msg}"
        
        # Step 7: Sort by name
        success, msg = op.execute("sort", {"column": "name", "ascending": True})
        assert success, f"Step 7 failed: {msg}"
        
        result = op.get_result()
        
        # Verify final state
        assert len(result) < original_rows  # Some rows were filtered
        assert result["name"].iloc[0] == "Alice"  # Sorted alphabetically
        assert result["email"].isna().sum() == 0  # No null emails
        assert result["score"].isna().sum() == 0  # No null scores
        assert (result["age"] >= 0).all() and (result["age"] <= 120).all()  # Valid ages
        
        # Verify operations log
        assert len(op.operations_log) == 8  # 8 operations executed

    def test_analysis_pipeline_with_aggregation(self, sample_dataset):
        """
        Test an analysis pipeline:
        1. Clean the data (drop nulls, filter invalid)
        2. Add a category column based on score
        3. Group by department and aggregate
        """
        op = DataOperator(sample_dataset)
        
        # Step 1: Drop rows with null age or score
        success, _ = op.execute("drop_nulls", {"column": "age"})
        assert success
        success, _ = op.execute("drop_nulls", {"column": "score"})
        assert success
        
        # Step 2: Add performance category based on score
        success, msg = op.execute("add_conditional_column", {
            "name": "performance",
            "condition_column": "score",
            "conditions": [
                {"operator": "<", "value": 70, "result": "Low"},
                {"operator": "between", "value1": 70, "value2": 85, "result": "Medium"},
                {"operator": ">", "value": 85, "result": "High"},
            ],
            "default_value": "Unknown"
        })
        assert success, f"Add conditional column failed: {msg}"
        assert "performance" in op.get_result().columns
        
        # Step 3: Group by department and calculate average score
        success, msg = op.execute("group_aggregate", {
            "group_by": "department",
            "aggregations": {"score": "mean", "age": "mean"}
        })
        assert success, f"Group aggregate failed: {msg}"
        
        result = op.get_result()
        
        # Should have one row per department
        assert len(result) <= 3  # Max 3 departments
        assert "department" in result.columns
        assert "score" in result.columns or "score_mean" in result.columns

    def test_validation_and_quarantine_pipeline(self, sample_dataset):
        """
        Test validation and quarantine workflow:
        1. Validate schema
        2. Quarantine null emails
        3. Quarantine out-of-range ages
        4. Quarantine invalid status values
        5. Verify quarantined rows are separated
        """
        op = DataOperator(sample_dataset)
        
        # Step 1: Validate schema (should fail due to data issues)
        success, msg = op.execute("validate_schema", {
            "columns": ["id", "name", "email", "age"],
            "not_null": ["email", "age"],
            "ranges": {"age": {"min": 0, "max": 120}}
        })
        assert success
        assert "FAILED" in msg  # Should fail due to nulls and out-of-range
        
        # Step 2: Quarantine null emails
        success, msg = op.execute("quarantine_rows", {
            "column": "email",
            "condition": "null"
        })
        assert success
        assert "Quarantined" in msg
        
        # Step 3: Quarantine out-of-range ages
        success, msg = op.execute("quarantine_rows", {
            "column": "age",
            "condition": "range",
            "min": 0,
            "max": 120
        })
        assert success
        
        # Step 4: Quarantine rows with ERROR status
        success, msg = op.execute("quarantine_rows", {
            "column": "status",
            "condition": "values",
            "values": ["ERROR", "INVALID"]
        })
        assert success
        
        result = op.get_result()
        quarantined = op.get_quarantined()
        
        # Verify main data is clean
        assert result["email"].isna().sum() == 0
        assert (result["age"] >= 0).all() and (result["age"] <= 120).all()
        assert not result["status"].isin(["ERROR", "INVALID"]).any()
        
        # Verify quarantined rows exist
        assert quarantined is not None
        assert len(quarantined) > 0
        assert "_quarantine_reason" in quarantined.columns

    def test_column_operations_pipeline(self, sample_dataset):
        """
        Test column manipulation pipeline:
        1. Keep only specific columns
        2. Rename a column
        3. Add a derived column
        4. Convert type
        5. Drop a column
        """
        op = DataOperator(sample_dataset)
        
        # Step 1: Keep only specific columns
        success, msg = op.execute("keep_columns", {
            "columns": ["id", "name", "age", "score", "department"]
        })
        assert success, f"Keep columns failed: {msg}"
        assert len(op.get_result().columns) == 5
        assert "email" not in op.get_result().columns
        
        # Step 2: Rename column
        success, msg = op.execute("rename_column", {
            "old_name": "department",
            "new_name": "dept"
        })
        assert success
        assert "dept" in op.get_result().columns
        assert "department" not in op.get_result().columns
        
        # Step 3: Add derived column (name length)
        success, msg = op.execute("add_column", {
            "name": "name_length",
            "from_column": "name",
            "operation": "length"
        })
        assert success
        assert "name_length" in op.get_result().columns
        
        # Step 4: Convert score to string
        success, msg = op.execute("convert_type", {
            "column": "score",
            "dtype": "str"
        })
        assert success
        
        # Step 5: Drop the name_length column
        success, msg = op.execute("drop_column", {"column": "name_length"})
        assert success
        assert "name_length" not in op.get_result().columns
        
        # Verify operations log
        assert len(op.operations_log) == 5

    def test_sampling_and_limit_operations(self, sample_dataset):
        """
        Test limit and sample operations:
        1. Limit to first N rows
        2. Sample random rows
        """
        # Test limit_rows
        op = DataOperator(sample_dataset)
        success, msg = op.execute("limit_rows", {"n": 5})
        assert success
        assert len(op.get_result()) == 5
        
        # Test limit from end
        op = DataOperator(sample_dataset)
        success, msg = op.execute("limit_rows", {"n": 3, "from_end": True})
        assert success
        assert len(op.get_result()) == 3
        
        # Test sample_rows with n
        op = DataOperator(sample_dataset)
        success, msg = op.execute("sample_rows", {"n": 4, "random_state": 42})
        assert success
        assert len(op.get_result()) == 4
        
        # Test sample_rows with fraction
        op = DataOperator(sample_dataset)
        success, msg = op.execute("sample_rows", {"fraction": 0.5, "random_state": 42})
        assert success
        assert len(op.get_result()) == 5  # 50% of 10

    def test_full_etl_pipeline(self, sample_dataset):
        """
        Test a complete ETL-like pipeline combining all primitive types:
        1. Validate input
        2. Clean data (strip, lowercase, fill nulls)
        3. Quarantine bad data
        4. Transform (add columns, filter)
        5. Aggregate
        6. Sort and limit output
        """
        op = DataOperator(sample_dataset)
        
        # 1. Validate
        success, msg = op.execute("validate_schema", {
            "columns": ["id", "name", "age", "score"],
            "types": {"id": "int", "age": "float"}
        })
        assert success
        
        # 2. Clean - strip whitespace
        success, _ = op.execute("strip_whitespace", {"column": "name"})
        assert success
        
        # 3. Clean - lowercase status
        success, _ = op.execute("lowercase", {"column": "status"})
        assert success
        
        # 4. Quarantine null ages
        success, _ = op.execute("quarantine_rows", {"column": "age", "condition": "null"})
        assert success
        
        # 5. Quarantine invalid ages
        success, _ = op.execute("quarantine_rows", {"column": "age", "condition": "range", "min": 0, "max": 120})
        assert success
        
        # 6. Fill null scores
        success, _ = op.execute("fill_nulls", {"column": "score", "method": "median"})
        assert success
        
        # 7. Add category column
        success, _ = op.execute("add_conditional_column", {
            "name": "age_group",
            "condition_column": "age",
            "conditions": [
                {"operator": "<", "value": 30, "result": "Young"},
                {"operator": "between", "value1": 30, "value2": 50, "result": "Middle"},
                {"operator": ">", "value": 50, "result": "Senior"}
            ],
            "default_value": "Unknown"
        })
        assert success
        
        # 8. Filter active only
        success, _ = op.execute("filter_rows", {"column": "status", "operator": "==", "value": "active"})
        assert success
        
        # 9. Sort by score descending
        success, _ = op.execute("sort", {"column": "score", "ascending": False})
        assert success
        
        # 10. Limit to top 5
        success, _ = op.execute("limit_rows", {"n": 5})
        assert success
        
        result = op.get_result()
        quarantined = op.get_quarantined()
        
        # Final assertions
        assert len(result) <= 5
        assert result["score"].isna().sum() == 0
        assert "age_group" in result.columns
        assert all(result["status"] == "active")
        
        # Quarantine should have captured bad rows
        assert quarantined is not None
        
        # All 10 operations logged
        assert len(op.operations_log) == 10
        
        # Get summary
        summary = op.get_summary()
        assert "Changes Applied" in summary
        assert "Operations (10)" in summary
