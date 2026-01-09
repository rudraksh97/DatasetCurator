"""Unified LangGraph-based workflow for dataset processing and transformations.

This module provides:
1. Dataset upload and initial processing
2. Multi-step transformations with LangGraph
3. State management and persistence
4. Conditional branching, retries, and validation

Refactored to use:
- Repository pattern for data access
- Centralized configuration
- Separated node implementations
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

import pandas as pd
from langgraph.graph import StateGraph, END
from sqlalchemy.ext.asyncio import AsyncSession

from agents.data_ops import DataOperator
from config import settings
from models.dataset_state import DatasetState
from repositories.dataset import DatasetRepository
from services.llm.planner import create_execution_plan
from services.storage import get_storage


# Re-export storage paths for backward compatibility (local storage only)
RAW_STORAGE = settings.storage.raw_path if not settings.storage.is_s3 else None
CURATED_STORAGE = settings.storage.curated_path if not settings.storage.is_s3 else None


# ============================================================================
# Step Status and Results
# ============================================================================

class StepStatus(str, Enum):
    """Status of a transformation step."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    NEEDS_APPROVAL = "needs_approval"


@dataclass
class StepResult:
    """Result of executing a single step.
    
    Attributes:
        step_num: Step number in the plan.
        description: Human-readable description.
        operation: Operation name executed.
        status: Execution status.
        message: Result or error message.
        rows_before: Row count before operation.
        rows_after: Row count after operation.
        retry_count: Number of retries attempted.
    """
    step_num: int
    description: str
    operation: str
    status: StepStatus
    message: str = ""
    rows_before: int = 0
    rows_after: int = 0
    retry_count: int = 0


# ============================================================================
# Workflow State
# ============================================================================

class TransformationState(TypedDict):
    """State passed between nodes in the transformation graph."""
    # Input
    user_message: str
    data_path: str
    columns: List[str]
    
    # Planning
    steps: List[Dict[str, Any]]
    current_step_idx: int
    is_analysis: bool  # True = read-only analysis (use copy), False = mutate dataset
    
    # Execution
    df: Any  # pd.DataFrame - using Any for TypedDict compatibility
    analysis_df: Any  # pd.DataFrame - copy for analysis operations
    results: List[StepResult]
    
    # Control flow
    needs_approval: bool
    approval_granted: bool
    error_message: str
    max_retries: int
    
    # Output
    final_message: str
    success: bool


# ============================================================================
# Database State Management (using Repository)
# ============================================================================

async def get_dataset_state(session: AsyncSession, dataset_id: str) -> DatasetState:
    """Get dataset state from database.
    
    Args:
        session: Database session.
        dataset_id: Dataset identifier.
    
    Returns:
        DatasetState instance.
    
    Raises:
        ValueError: If dataset not found.
    """
    repo = DatasetRepository(session)
    return await repo.get(dataset_id)


async def upsert_dataset_state(session: AsyncSession, state: DatasetState) -> DatasetState:
    """Create or update dataset state in database.
    
    Args:
        session: Database session.
        state: State to save.
    
    Returns:
        Saved state.
    """
    repo = DatasetRepository(session)
    return await repo.save(state)


# ============================================================================
# Upload Pipeline
# ============================================================================

async def process_upload(
    session: AsyncSession,
    dataset_id: str,
    source_path,
) -> DatasetState:
    """Process an uploaded dataset: analyze schema and create curated copy.
    
    Args:
        session: Database session.
        dataset_id: Unique dataset identifier.
        source_path: Path to uploaded file (Path for local, str key for S3).
    
    Returns:
        Created DatasetState.
    
    Raises:
        ValueError: If processing fails.
    """
    import io
    
    storage = get_storage()
    state = DatasetState(dataset_id=dataset_id, raw_path=str(source_path))
    
    # Read data for schema analysis
    try:
        if settings.storage.is_s3:
            content = await storage.read_file(str(source_path))
            df = pd.read_csv(io.BytesIO(content), nrows=2000)
        else:
            df = pd.read_csv(source_path, nrows=2000)
        state.schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    except Exception as e:
        print(f"[Upload] Warning: Could not analyze schema for {dataset_id}: {e}")
        state.schema = None

    # Create curated copy
    try:
        if settings.storage.is_s3:
            content = await storage.read_file(str(source_path))
            df_full = pd.read_csv(io.BytesIO(content))
            curated_key = f"{settings.storage.curated_prefix}/{dataset_id}_v1.csv"
            await storage.write_csv(curated_key, df_full)
            state.curated_path = curated_key
        else:
            curated_path = settings.storage.curated_path / f"{dataset_id}_v1.csv"
            df_full = pd.read_csv(source_path)
            df_full.to_csv(curated_path, index=False)
            state.curated_path = str(curated_path)
        state.current_version = 1
    except Exception as e:
        print(f"[Upload] Error: Could not create curated copy for {dataset_id}: {e}")
        raise ValueError(f"Failed to process dataset: {e}") from e

    repo = DatasetRepository(session)
    return await repo.save(state)


# ============================================================================
# LangGraph Transformation Nodes
# ============================================================================

async def plan_node(state: TransformationState) -> TransformationState:
    """Create execution plan from user message."""
    plan = await create_execution_plan(state["user_message"], state["columns"])
    steps = plan.get("steps", [])
    plan_error = plan.get("error", "")
    is_analysis = plan.get("is_analysis", False)
    
    if plan_error:
        error_message = f"Planning failed: {plan_error}"
    elif not steps:
        error_message = "Could not create execution plan - no steps were generated"
    else:
        error_message = ""
    
    # If analysis mode, prepare analysis_df as a copy
    analysis_df = None
    if is_analysis and state.get("df") is not None:
        analysis_df = state["df"].copy()
    
    return {
        **state,
        "steps": steps,
        "current_step_idx": 0,
        "is_analysis": is_analysis,
        "analysis_df": analysis_df,
        "results": [],
        "error_message": error_message,
    }


async def load_data_node(state: TransformationState) -> TransformationState:
    """Load DataFrame from path (supports both local and S3)."""
    import io
    
    try:
        storage = get_storage()
        # Ensure path is a string for storage backend
        data_path = str(state["data_path"])
        
        # Read from storage (works for both S3 and local)
        content = await storage.read_file(data_path)
        df = pd.read_csv(io.BytesIO(content))
        
        return {**state, "df": df, "error_message": ""}
    except Exception as e:
        return {**state, "df": None, "error_message": f"Failed to load data: {str(e)}"}


def check_approval_node(state: TransformationState) -> TransformationState:
    """Check if current step needs human approval (destructive operations)."""
    if state["current_step_idx"] >= len(state["steps"]):
        return state
    
    # If already approved, we can proceed
    if state.get("approval_granted"):
        return {**state, "needs_approval": False}

    # Analysis operations are safe (read-only views)
    if state.get("is_analysis"):
        return {**state, "needs_approval": False}
    
    # Any permanent change (non-analysis) requires approval
    return {**state, "needs_approval": True}


def _create_step_result(
    step_num: int,
    description: str,
    operation: str,
    status: StepStatus,
    message: str,
    rows_before: int = 0,
    rows_after: int = 0,
    retry_count: int = 0,
) -> StepResult:
    """Create a StepResult with the given parameters."""
    return StepResult(
        step_num=step_num,
        description=description,
        operation=operation,
        status=status,
        message=message,
        rows_before=rows_before,
        rows_after=rows_after,
        retry_count=retry_count,
    )


def _execute_operation(
    df: pd.DataFrame, 
    operation: str, 
    params: Dict[str, Any],
) -> tuple[bool, str, Optional[pd.DataFrame]]:
    """Execute a data operation and return (success, message, result_df)."""
    operator = DataOperator(df)
    success, msg = operator.execute(operation, params)
    return success, msg, operator.get_result() if success else None


def execute_step_node(state: TransformationState) -> TransformationState:
    """Execute the current step."""
    idx = state["current_step_idx"]
    if idx >= len(state["steps"]):
        return state
    
    step = state["steps"][idx]
    step_num = step.get("step", idx + 1)
    description = step.get("description", "Unknown step")
    operation = step.get("operation")
    params = step.get("params", {})
    
    df = state["df"]
    results = list(state["results"])
    is_analysis = state.get("is_analysis", False)
    
    if df is None:
        result = _create_step_result(step_num, description, operation or "unknown", StepStatus.FAILED, "No data loaded")
        return {**state, "results": results + [result], "error_message": "No data loaded"}
    
    if not operation:
        result = _create_step_result(step_num, description, "unknown", StepStatus.SKIPPED, "No operation specified")
        return {**state, "results": results + [result], "current_step_idx": idx + 1}
    
    # Use analysis_df if in analysis mode, otherwise use main df
    working_df = state.get("analysis_df") if is_analysis else df
    if working_df is None:
        working_df = df
    
    rows_before = len(working_df)
    
    try:
        success, msg, new_df = _execute_operation(working_df, operation, params)
        
        if success and new_df is not None:
            result = _create_step_result(
                step_num, description, operation, StepStatus.SUCCESS, msg,
                rows_before=rows_before, rows_after=len(new_df)
            )

            # If analysis mode, update analysis_df and keep original df intact
            # If transformation mode, update df directly
            if is_analysis:
                return {
                    **state,
                    "analysis_df": new_df,
                    "df": df,  # original unchanged
                    "results": results + [result],
                    "current_step_idx": idx + 1,
                    "error_message": "",
                }
            else:
                return {
                    **state,
                    "df": new_df,
                    "results": results + [result],
                    "current_step_idx": idx + 1,
                    "error_message": "",
                }
        
        # Operation failed - track retry count
        retry_count = 0
        updated_results = list(results)
        for r in updated_results:
            if r.step_num == step_num:
                retry_count = r.retry_count + 1
                updated_results.remove(r)
                break
        
        result = _create_step_result(
            step_num, description, operation, StepStatus.FAILED, msg,
            rows_before=rows_before, retry_count=retry_count
        )
        return {**state, "results": updated_results + [result], "error_message": msg}
        
    except Exception as e:
        result = _create_step_result(
            step_num, description, operation, StepStatus.FAILED, str(e),
            rows_before=rows_before
        )
        return {**state, "results": results + [result], "error_message": str(e)}


def validate_step_node(state: TransformationState) -> TransformationState:
    """Validate the result of the last step."""
    if not state["results"]:
        return state
    
    last_result = state["results"][-1]
    is_analysis = state.get("is_analysis", False)
    
    if last_result.status == StepStatus.SUCCESS:
        # Skip warnings for analysis mode (operations on a copy)
        if not is_analysis:
            df = state["df"]
            
            if df is not None:
                if last_result.rows_before > 0:
                    removal_rate = 1 - (last_result.rows_after / last_result.rows_before)
                    if removal_rate > 0.9:
                        last_result.message += f" ⚠️ Warning: Removed {removal_rate*100:.1f}% of rows"
                
                if len(df) == 0:
                    last_result.message += " ⚠️ Warning: Dataset is now empty"
    
    return state


def finalize_node(state: TransformationState) -> TransformationState:
    """Create final summary message."""
    results = state["results"]
    df = state["df"]
    is_analysis = state.get("is_analysis", False)
    analysis_df = state.get("analysis_df")
    
    # If approval is required but not granted, short-circuit with a clear message
    if state.get("needs_approval") and not state.get("approval_granted"):
        idx = state.get("current_step_idx", 0)
        step = state["steps"][idx] if idx < len(state["steps"]) else {}
        step_num = step.get("step", idx + 1)
        description = step.get("description", "destructive operation")
        op = step.get("operation", "unknown")
        msg = (
            f"⚠️ **Approval required** before executing step {step_num} "
            f"(`{op}`): {description}.\n\n"
            "This operation may delete or modify a large number of rows. "
            "Use the **Approve change** button in the UI to continue, or ignore this "
            "message to cancel."
        )
        return {
            **state,
            "final_message": msg,
            "success": False,
        }
    
    # Use analysis_df for reporting if in analysis mode, otherwise use df
    result_df = analysis_df if is_analysis and analysis_df is not None else df
    
    if not state["steps"] and state["error_message"]:
        return {
            **state,
            "final_message": f"**Error:** {state['error_message']}",
            "success": False,
        }
    
    successful = sum(1 for r in results if r.status == StepStatus.SUCCESS)
    failed = sum(1 for r in results if r.status == StepStatus.FAILED)
    total = len(state["steps"])
    
    parts = [f"**Executed {total} step{'s' if total != 1 else ''}:**\n"]
    
    for result in results:
        icon = {
            StepStatus.SUCCESS: "✓",
            StepStatus.FAILED: "✗",
            StepStatus.SKIPPED: "⊘",
            StepStatus.NEEDS_APPROVAL: "⏸",
        }.get(result.status, "?")
        
        parts.append(f"\n**Step {result.step_num}:** {result.description}")
        parts.append(f"\n  {icon} {result.message}")
        
        if result.retry_count > 0:
            parts.append(f" (retried {result.retry_count}x)")
    
    parts.append(f"\n\n**Summary:** {successful}/{total} steps completed")
    
    if failed > 0:
        parts.append(f" ({failed} failed)")
    
    if result_df is not None and len(result_df) > 0:
        parts.append(f"\n**Result:** {len(result_df)} rows × {len(result_df.columns)} columns")
        if is_analysis:
            parts.append(" (analysis-only, original dataset unchanged)")
    
    return {
        **state,
        "final_message": "".join(parts),
        "success": failed == 0 and successful > 0,
    }


# ============================================================================
# Conditional Edge Functions
# ============================================================================

def should_retry_or_continue(state: TransformationState) -> Literal["retry", "continue", "finalize"]:
    """Decide whether to retry, continue, or finish."""
    # If no steps were generated or all steps done, finalize
    if not state["steps"] or state["current_step_idx"] >= len(state["steps"]):
        return "finalize"
    
    # If no results yet but we have steps, continue to execute
    if not state["results"]:
        return "continue"
    
    last_result = state["results"][-1]
    
    if last_result.status == StepStatus.SUCCESS:
        if state["current_step_idx"] >= len(state["steps"]):
            return "finalize"
        return "continue"
    
    elif last_result.status == StepStatus.FAILED:
        if last_result.retry_count < state["max_retries"]:
            return "retry"
        state["current_step_idx"] = state.get("current_step_idx", 0) + 1
        if state["current_step_idx"] >= len(state["steps"]):
            return "finalize"
        return "continue"
    
    return "continue"


def needs_approval_check(state: TransformationState) -> Literal["wait_approval", "execute", "finalize"]:
    """Check if we need approval or if we should finalize."""
    # If no steps or all steps done, go to finalize
    if not state["steps"] or state["current_step_idx"] >= len(state["steps"]):
        return "finalize"
    
    if state["needs_approval"] and not state["approval_granted"]:
        return "wait_approval"
    return "execute"


# ============================================================================
# Build Transformation Graph
# ============================================================================

def create_transformation_graph() -> StateGraph:
    """Create the LangGraph workflow for dataset transformations.
    
    Returns:
        Compiled StateGraph ready for execution.
    """
    graph = StateGraph(TransformationState)
    
    graph.add_node("plan", plan_node)
    graph.add_node("load_data", load_data_node)
    graph.add_node("check_approval", check_approval_node)
    graph.add_node("execute_step", execute_step_node)
    graph.add_node("validate", validate_step_node)
    graph.add_node("finalize", finalize_node)
    
    graph.set_entry_point("plan")
    graph.add_edge("plan", "load_data")
    graph.add_edge("load_data", "check_approval")
    
    graph.add_conditional_edges(
        "check_approval",
        needs_approval_check,
        {"execute": "execute_step", "wait_approval": "finalize", "finalize": "finalize"}
    )
    
    graph.add_edge("execute_step", "validate")
    
    graph.add_conditional_edges(
        "validate",
        should_retry_or_continue,
        {"retry": "execute_step", "continue": "check_approval", "finalize": "finalize"}
    )
    
    graph.add_edge("finalize", END)
    
    return graph.compile()


# ============================================================================
# Workflow Executors
# ============================================================================

async def execute_transformation(
    user_message: str,
    data_path: str,
    columns: List[str],
    max_retries: int = 1,
    approval_granted: bool = False,
) -> tuple[bool, str, Optional[pd.DataFrame], bool, bool]:
    """Execute transformation workflow and return result.
    
    Args:
        user_message: User's transformation request.
        data_path: Path to data file.
        columns: Available column names.
        max_retries: Maximum retry attempts per step.
    
    Returns:
        Tuple of (success, final_message, result_dataframe).
    """
    graph = create_transformation_graph()
    
    initial_state: TransformationState = {
        "user_message": user_message,
        "data_path": data_path,
        "columns": columns,
        "steps": [],
        "current_step_idx": 0,
        "is_analysis": False,
        "df": None,
        "analysis_df": None,
        "results": [],
        "needs_approval": False,
        "approval_granted": approval_granted,
        "error_message": "",
        "max_retries": max_retries,
        "final_message": "",
        "success": False,
    }
    
    final_state = await graph.ainvoke(initial_state)
    
    # Return analysis_df for preview if in analysis mode, otherwise return df
    is_analysis = final_state.get("is_analysis", False)
    steps = final_state.get("steps", [])
    result_df = final_state.get("analysis_df") if is_analysis else final_state["df"]
    
    # For analysis mode, include both for the handler to decide
    return (
        final_state["success"],
        final_state["final_message"],
        result_df if result_df is not None else final_state["df"],
        is_analysis
    )
