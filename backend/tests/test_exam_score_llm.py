"""Integration tests for LLM pipeline using Exam Score Prediction dataset."""
import os
import pytest
import pandas as pd
from pathlib import Path

from agents.data_ops import DataOperator
from services.llm.planner import ExecutionPlannerService

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set - skipping LLM integration tests"
)

@pytest.fixture
def exam_score_dataset():
    """Load the Exam Score Prediction dataset."""
    # Assuming the test is run from project root or backend dir
    # Try multiple common paths
    paths = [
        "examples/Exam_Score_Prediction.csv",
        "../examples/Exam_Score_Prediction.csv",
        "/Applications/GitHub/DatasetCurator/examples/Exam_Score_Prediction.csv"
    ]
    
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    
    pytest.fail(f"Could not find Exam_Score_Prediction.csv in {paths}")

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

class TestExamScoreLLM:
    """Tests for Exam Score Prediction dataset using LLM planner."""

    @pytest.mark.asyncio
    async def test_group_by_course_and_count(self, exam_score_dataset):
        """Test grouping by course and counting students."""
        planner = ExecutionPlannerService()
        columns = list(exam_score_dataset.columns)
        
        user_msg = "Group by course and count the number of students"
        print(f"\nUser message: {user_msg}")
        
        plan = await planner.create_plan(
            user_message=user_msg,
            columns=columns
        )
        
        assert "steps" in plan
        assert len(plan["steps"]) > 0
        
        # Execute
        op = await execute_plan(exam_score_dataset, plan)
        result = op.get_result()
        
        print(f"Result columns: {result.columns}")
        print(result.head())
        
        # Verify
        assert "course" in result.columns or result.index.name == "course"
        # Since 'count' operation usually keeps the grouping column or puts it in index
        # And there should be some count column. DataOperator usually standardizes this.
        # Let's check if the number of rows matches unique courses
        unique_courses = exam_score_dataset["course"].nunique()
        assert len(result) == unique_courses, f"Expected {unique_courses} rows, got {len(result)}"

    @pytest.mark.asyncio
    async def test_average_study_hours_by_course(self, exam_score_dataset):
        """Test calculating average study hours by course."""
        planner = ExecutionPlannerService()
        columns = list(exam_score_dataset.columns)
        
        user_msg = "Group by course and calculate the average study_hours"
        print(f"\nUser message: {user_msg}")
        
        plan = await planner.create_plan(
            user_message=user_msg,
            columns=columns
        )
        
        assert "steps" in plan
        assert len(plan["steps"]) > 0, f"Plan should not be empty for: {user_msg}"

        # Execute
        op = await execute_plan(exam_score_dataset, plan)
        result = op.get_result()
        
        print(f"Result columns: {result.columns}")
        print(result.head())
        
        # Verify
        assert "course" in result.columns or result.index.name == "course"
        
        # Verify row count matches courses
        unique_courses = exam_score_dataset["course"].nunique()
        assert len(result) == unique_courses

