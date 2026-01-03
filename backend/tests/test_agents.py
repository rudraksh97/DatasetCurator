"""Tests for data operations and dataset fetcher."""
import pandas as pd
import pytest
from agents.data_ops import DataOperator
from agents.dataset_fetcher import list_available_datasets, SAMPLE_DATASETS


def test_data_operator_drop_column():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    op = DataOperator(df)
    success, msg = op.execute("drop_column", {"column": "b"})
    assert success
    assert "b" not in op.get_result().columns


def test_data_operator_rename_column():
    df = pd.DataFrame({"old_name": [1, 2, 3]})
    op = DataOperator(df)
    success, msg = op.execute("rename_column", {"old_name": "old_name", "new_name": "new_name"})
    assert success
    assert "new_name" in op.get_result().columns
    assert "old_name" not in op.get_result().columns


def test_data_operator_filter_rows():
    df = pd.DataFrame({"age": [10, 20, 30, 40]})
    op = DataOperator(df)
    success, msg = op.execute("filter_rows", {"column": "age", "operator": ">", "value": 25})
    assert success
    assert len(op.get_result()) == 2


def test_data_operator_add_conditional_column():
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
    df = pd.DataFrame({"a": [1, 2, 3]})
    op = DataOperator(df)
    success, msg = op.execute("drop_column", {"column": "nonexistent"})
    assert not success
    assert "not found" in msg.lower()


def test_list_available_datasets():
    result = list_available_datasets()
    assert "titanic" in result.lower()
    assert "iris" in result.lower()


def test_sample_datasets_structure():
    """Verify all sample datasets have required fields."""
    for name, info in SAMPLE_DATASETS.items():
        assert "url" in info, f"{name} missing url"
        assert "description" in info, f"{name} missing description"
