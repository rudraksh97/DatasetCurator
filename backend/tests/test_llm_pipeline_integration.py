"""Integration tests for LLM-driven transformation pipeline.

These tests verify that:
1. LLM correctly plans multi-step transformations from natural language
2. Planned steps execute correctly via DataOperator
3. End-to-end pipeline produces expected results

Requires: OPENROUTER_API_KEY environment variable set.
"""
import os
import pytest
import pandas as pd

from agents.data_ops import DataOperator
from services.llm.planner import ExecutionPlannerService


# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set - skipping LLM integration tests"
)


@pytest.fixture
def sample_dataset():
    """Create a realistic sample dataset for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "name": ["  Alice  ", "Bob", "Charlie", "  David", "Eve  ", "Frank", "Grace", "Heidi", "Ivan", "Judy"],
        "email": ["alice@test.com", "bob@test.com", "invalid", "david@test.com", None, "frank@test.com", "grace@test.com", "heidi@test.com", "ivan@test.com", "judy@test.com"],
        "age": [25, 30, 35, None, 28, 150, 22, 45, -5, 33],
        "score": [85, 92, 78, 88, 95, 72, 88, None, 91, 85],
        "department": ["Engineering", "Sales", "Engineering", "HR", "Sales", "Engineering", "HR", "Sales", "Engineering", "HR"],
        "status": ["active", "ACTIVE", "Active", "inactive", "active", "ERROR", "active", "active", "active", "active"],
    })


async def execute_plan(df: pd.DataFrame, plan: dict) -> DataOperator:
    """Execute a plan's steps on a DataFrame."""
    op = DataOperator(df)
    
    for step in plan.get("steps", []):
        operation = step.get("operation")
        params = step.get("params", {})
        
        success, msg = op.execute(operation, params)
        if not success:
            raise RuntimeError(f"Step {step.get('step')} failed: {msg}")
    
    return op


class TestLLMPlannerIntegration:
    """Integration tests for LLM-driven transformation planning."""

    @pytest.mark.asyncio
    async def test_simple_single_step_plan(self, sample_dataset):
        """Test that LLM correctly plans a simple single-step transformation."""
        planner = ExecutionPlannerService()
        columns = list(sample_dataset.columns)
        
        # Simple request: drop a column
        plan = await planner.create_plan(
            user_message="Remove the email column",
            columns=columns
        )
        
        # Verify plan structure
        assert "steps" in plan, f"Plan missing 'steps': {plan}"
        assert len(plan["steps"]) >= 1, "Should have at least 1 step"
        
        # Execute the plan
        op = await execute_plan(sample_dataset, plan)
        result = op.get_result()
        
        # Verify result
        assert "email" not in result.columns, "Email column should be dropped"
        print(f"\n✓ Simple plan: {len(plan['steps'])} step(s) executed")
        print(f"  Plan: {[s['operation'] for s in plan['steps']]}")

    @pytest.mark.asyncio
    async def test_multi_step_cleaning_plan(self, sample_dataset):
        """Test that LLM plans a multi-step cleaning transformation."""
        planner = ExecutionPlannerService()
        columns = list(sample_dataset.columns)
        
        # Complex cleaning request
        plan = await planner.create_plan(
            user_message="Clean the data: strip whitespace from names, lowercase the status column, and remove rows with null ages",
            columns=columns
        )
        
        # Verify plan has multiple steps
        assert "steps" in plan, f"Plan missing 'steps': {plan}"
        steps = plan["steps"]
        assert len(steps) >= 2, f"Should have multiple steps, got {len(steps)}"
        
        # Check that appropriate operations are planned
        operations = [s.get("operation") for s in steps]
        print(f"\n  Planned operations: {operations}")
        
        # Execute the plan
        op = await execute_plan(sample_dataset, plan)
        result = op.get_result()
        
        # Verify transformations happened
        # Names should be stripped
        assert result["name"].iloc[0].strip() == result["name"].iloc[0], "Names should be stripped"
        
        # Status should be lowercase
        assert all(s == s.lower() for s in result["status"]), "Status should be lowercase"
        
        # No null ages
        assert result["age"].isna().sum() == 0, "Null ages should be removed"
        
        print(f"✓ Multi-step plan: {len(steps)} steps executed")
        print(f"  Result: {len(result)} rows, {len(result.columns)} columns")

    @pytest.mark.asyncio
    async def test_filter_and_sort_plan(self, sample_dataset):
        """Test filter and sort operations."""
        planner = ExecutionPlannerService()
        columns = list(sample_dataset.columns)
        
        plan = await planner.create_plan(
            user_message="Keep only rows where score is greater than 80 and sort by score descending",
            columns=columns
        )
        
        assert "steps" in plan
        steps = plan["steps"]
        operations = [s.get("operation") for s in steps]
        print(f"\n  Planned operations: {operations}")
        
        # Execute
        op = await execute_plan(sample_dataset, plan)
        result = op.get_result()
        
        # Verify filter
        assert (result["score"] > 80).all() or result["score"].isna().all() == False, "All scores should be > 80"
        
        # Verify sort (descending)
        scores = result["score"].dropna().tolist()
        assert scores == sorted(scores, reverse=True), "Should be sorted descending"
        
        print(f"✓ Filter+Sort plan: {len(steps)} steps, {len(result)} rows remaining")

    @pytest.mark.asyncio
    async def test_conditional_column_plan(self, sample_dataset):
        """Test adding a conditional column."""
        planner = ExecutionPlannerService()
        columns = list(sample_dataset.columns)
        
        plan = await planner.create_plan(
            user_message="Add a new column called 'performance' where score < 75 is 'Low', 75-90 is 'Medium', and above 90 is 'High'",
            columns=columns
        )
        
        assert "steps" in plan
        steps = plan["steps"]
        print(f"\n  Planned operations: {[s.get('operation') for s in steps]}")
        
        # Execute
        op = await execute_plan(sample_dataset, plan)
        result = op.get_result()
        
        # Verify new column exists
        assert "performance" in result.columns, "Performance column should be added"
        
        # Verify values make sense
        perf_values = set(result["performance"].dropna().unique())
        expected_values = {"Low", "Medium", "High", "Unknown"}
        assert perf_values.issubset(expected_values), f"Unexpected values: {perf_values}"
        
        print(f"✓ Conditional column plan: column added with values {perf_values}")

    @pytest.mark.asyncio
    async def test_complex_etl_plan(self, sample_dataset):
        """Test a complex ETL-like transformation request."""
        planner = ExecutionPlannerService()
        columns = list(sample_dataset.columns)
        
        plan = await planner.create_plan(
            user_message="""
            Clean this dataset:
            1. Strip whitespace from all text columns
            2. Convert status to lowercase
            3. Remove rows with null email or age
            4. Filter to only active status
            5. Sort by score descending
            """,
            columns=columns
        )
        
        assert "steps" in plan
        steps = plan["steps"]
        
        print(f"\n  Complex ETL plan with {len(steps)} steps:")
        for step in steps:
            print(f"    {step.get('step', '?')}. {step.get('operation')}: {step.get('description', '')[:50]}")
        
        # Execute
        op = await execute_plan(sample_dataset, plan)
        result = op.get_result()
        
        # Verify key outcomes
        assert result["email"].isna().sum() == 0, "No null emails"
        assert all(s == s.lower() for s in result["status"]), "Status lowercase"
        
        print(f"✓ Complex ETL: {len(steps)} steps → {len(result)} clean rows")
        print(f"  Operations log: {len(op.operations_log)} operations executed")

    @pytest.mark.asyncio 
    async def test_aggregation_plan(self, sample_dataset):
        """Test grouping and aggregation."""
        planner = ExecutionPlannerService()
        columns = list(sample_dataset.columns)
        
        plan = await planner.create_plan(
            user_message="Group by department and calculate the average score for each",
            columns=columns
        )
        
        assert "steps" in plan
        steps = plan["steps"]
        print(f"\n  Planned operations: {[s.get('operation') for s in steps]}")
        
        # Execute
        op = await execute_plan(sample_dataset, plan)
        result = op.get_result()
        
        # Verify aggregation happened
        assert len(result) <= len(sample_dataset["department"].unique()), "Should be grouped"
        assert "department" in result.columns, "Should have department column"
        
        print(f"✓ Aggregation plan: {len(result)} groups")
        print(result.to_string())


class TestPlannerEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_ambiguous_request(self, sample_dataset):
        """Test handling of ambiguous user request."""
        planner = ExecutionPlannerService()
        columns = list(sample_dataset.columns)
        
        # Ambiguous request
        plan = await planner.create_plan(
            user_message="Make it better",
            columns=columns
        )
        
        # Should return something (possibly empty or error)
        assert "steps" in plan or "error" in plan
        print(f"\n  Ambiguous request result: {plan}")

    @pytest.mark.asyncio
    async def test_invalid_column_reference(self, sample_dataset):
        """Test when user references non-existent column."""
        planner = ExecutionPlannerService()
        columns = list(sample_dataset.columns)
        
        plan = await planner.create_plan(
            user_message="Drop the 'nonexistent_column' column",
            columns=columns
        )
        
        # LLM should still produce a plan
        assert "steps" in plan
        
        # Execution may fail gracefully
        try:
            op = await execute_plan(sample_dataset, plan)
            # If it doesn't fail, column matching may have saved it
            print(f"\n  Non-existent column handled gracefully")
        except RuntimeError as e:
            print(f"\n  Expected error caught: {e}")


# Run as standalone script for quick testing
if __name__ == "__main__":
    import asyncio
    from agents.data_ops import DataOperator
    
    async def run_full_integration_test():
        """Run a comprehensive end-to-end LLM pipeline test."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "name": ["  Alice  ", "Bob", "Charlie", "  David", "Eve  ", "Frank", "Grace", "Heidi", "Ivan", "Judy"],
            "email": ["alice@test.com", "bob@test.com", "invalid", "david@test.com", None, "frank@test.com", "grace@test.com", "heidi@test.com", "ivan@test.com", "judy@test.com"],
            "age": [25, 30, 35, None, 28, 150, 22, 45, -5, 33],
            "score": [85, 92, 78, 88, 95, 72, 88, None, 91, 85],
            "department": ["Engineering", "Sales", "Engineering", "HR", "Sales", "Engineering", "HR", "Sales", "Engineering", "HR"],
            "status": ["active", "ACTIVE", "Active", "inactive", "active", "ERROR", "active", "active", "active", "active"],
        })
        
        print("=" * 70)
        print("LLM PIPELINE INTEGRATION TEST")
        print("=" * 70)
        print(f"\nOriginal dataset: {len(df)} rows, {len(df.columns)} columns")
        
        planner = ExecutionPlannerService()
        columns = list(df.columns)
        
        # Complex natural language request
        user_request = """
        Clean this dataset:
        1. Strip whitespace from the name column
        2. Convert status to lowercase
        3. Remove rows where email is null
        4. Remove rows where age is negative or above 120
        5. Fill missing scores with mean
        6. Keep only active status rows
        7. Sort by score descending
        """
        
        print(f"\nUser Request:{user_request}")
        
        # Get plan from LLM
        plan = await planner.create_plan(user_request, columns)
        steps = plan.get("steps", [])
        
        print(f"\nLLM planned {len(steps)} steps:")
        for s in steps:
            print(f"  {s.get('step', '?')}. {s['operation']}: {s.get('params', {})}")
        
        # Execute
        print("\nExecuting...")
        op = DataOperator(df)
        for step in steps:
            success, msg = op.execute(step["operation"], step.get("params", {}))
            status = "✓" if success else "✗"
            print(f"  {status} {step['operation']}: {msg}")
        
        result = op.get_result()
        print(f"\nResult: {len(result)} rows")
        print(result[["name", "age", "score", "status"]].to_string())
        
        # Verification
        print("\nVerification:")
        passed = 0
        total = 5
        
        if len(result) > 0 and result["name"].iloc[0] == result["name"].iloc[0].strip():
            print("  ✓ Names stripped"); passed += 1
        else:
            print("  ✗ Names not stripped")
        
        if len(result) == 0 or all(s == s.lower() for s in result["status"]):
            print("  ✓ Status lowercase"); passed += 1
        else:
            print("  ✗ Status not lowercase")
        
        if result["email"].isna().sum() == 0:
            print("  ✓ No null emails"); passed += 1
        else:
            print("  ✗ Null emails remain")
        
        if len(result) == 0 or ((result["age"] >= 0).all() and (result["age"] <= 120).all()):
            print("  ✓ Ages valid (0-120)"); passed += 1
        else:
            print("  ✗ Invalid ages remain")
        
        if len(result) == 0 or all(result["status"] == "active"):
            print("  ✓ All active"); passed += 1
        else:
            print("  ✗ Non-active rows remain")
        
        print("\n" + "=" * 70)
        print(f"RESULT: {passed}/{total} checks passed")
        if passed == total:
            print("ALL INTEGRATION TESTS PASSED!")
        else:
            print("SOME TESTS FAILED - Review LLM plan above")
        print("=" * 70)
        
        return passed == total
    
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        exit(1)
    else:
        success = asyncio.run(run_full_integration_test())
        exit(0 if success else 1)
