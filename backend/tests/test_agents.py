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
