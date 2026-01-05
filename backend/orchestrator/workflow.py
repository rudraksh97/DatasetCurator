"""Unified LangGraph-based workflow for dataset processing and transformations.

This module provides:
1. Dataset upload and initial processing
2. Multi-step transformations with LangGraph
3. State management and persistence
4. Conditional branching, retries, and validation
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
from llm import create_execution_plan
from models.dataset_state import DatasetState
from models.db_models import DatasetRecord


# ============================================================================
# Storage Configuration
# ============================================================================

RAW_STORAGE = Path("storage/raw")
CURATED_STORAGE = Path("storage/curated")

# Ensure directories exist
RAW_STORAGE.mkdir(parents=True, exist_ok=True)
CURATED_STORAGE.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Step Status and Results
# ============================================================================

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    NEEDS_APPROVAL = "needs_approval"


@dataclass
class StepResult:
    """Result of executing a single step."""
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
    
    # Execution
    df: Any  # pd.DataFrame - using Any for TypedDict compatibility
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
# Database State Management
# ============================================================================

async def get_dataset_state(session: AsyncSession, dataset_id: str) -> DatasetState:
    """Get dataset state from database."""
    record = await session.get(DatasetRecord, dataset_id)
    if not record:
        raise ValueError(f"Dataset {dataset_id} not found")
    return DatasetState.from_record(record)


async def upsert_dataset_state(session: AsyncSession, state: DatasetState) -> DatasetState:
    """Create or update dataset state in database."""
    payload = state.to_record_payload()
    record = await session.get(DatasetRecord, state.dataset_id)
    if record:
        for key, value in payload.items():
            setattr(record, key, value)
    else:
        record = DatasetRecord(**payload)
        session.add(record)
    await session.commit()
    await session.refresh(record)
    return DatasetState.from_record(record)


# ============================================================================
# Upload Pipeline
# ============================================================================

async def process_upload(
    session: AsyncSession,
    dataset_id: str,
    source_path: Path,
) -> DatasetState:
    """Process an uploaded dataset: analyze schema and create curated copy."""
    
    # 1. Create initial state
    state = DatasetState(dataset_id=dataset_id, raw_path=str(source_path))
    
    # 2. Read and analyze the data
    try:
        df = pd.read_csv(source_path, nrows=2000)
        state.schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    except Exception:
        pass  # Schema will remain None on error

    # 3. Create curated copy
    curated_path = CURATED_STORAGE / f"{dataset_id}_v1.csv"
    try:
        df_full = pd.read_csv(source_path)
        df_full.to_csv(curated_path, index=False)
        state.curated_path = str(curated_path)
        state.current_version = 1
    except Exception:
        state.curated_path = None

    # 4. Save to database
    state = await upsert_dataset_state(session, state)
    return state


# ============================================================================
# LangGraph Transformation Nodes
# ============================================================================

async def plan_node(state: TransformationState) -> TransformationState:
    """Create execution plan from user message."""
    plan = await create_execution_plan(state["user_message"], state["columns"])
    steps = plan.get("steps", [])
    
    return {
        **state,
        "steps": steps,
        "current_step_idx": 0,
        "results": [],
        "error_message": "" if steps else "Could not create execution plan",
    }


def load_data_node(state: TransformationState) -> TransformationState:
    """Load DataFrame from path."""
    try:
        df = pd.read_csv(state["data_path"])
        return {
            **state,
            "df": df,
            "error_message": "",
        }
    except Exception as e:
        return {
            **state,
            "df": None,
            "error_message": f"Failed to load data: {str(e)}",
        }


def check_approval_node(state: TransformationState) -> TransformationState:
    """Check if current step needs human approval (destructive operations)."""
    if state["current_step_idx"] >= len(state["steps"]):
        return state
    
    step = state["steps"][state["current_step_idx"]]
    operation = step.get("operation", "")
    
    # Destructive operations that might need approval
    destructive_ops = ["drop_column", "drop_rows", "drop_nulls", "drop_duplicates"]
    
    needs_approval = False
    if operation in destructive_ops:
        df = state["df"]
        if df is not None and len(df) > 1000:
            # Large dataset + destructive operation
            # Set to True to enable approval flow
            needs_approval = False
    
    return {
        **state,
        "needs_approval": needs_approval,
    }


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
    
    if df is None:
        result = StepResult(
            step_num=step_num,
            description=description,
            operation=operation or "unknown",
            status=StepStatus.FAILED,
            message="No data loaded",
        )
        results.append(result)
        return {**state, "results": results, "error_message": "No data loaded"}
    
    if not operation:
        result = StepResult(
            step_num=step_num,
            description=description,
            operation="unknown",
            status=StepStatus.SKIPPED,
            message="No operation specified",
        )
        results.append(result)
        return {**state, "results": results, "current_step_idx": idx + 1}
    
    rows_before = len(df)
    
    try:
        operator = DataOperator(df)
        success, msg = operator.execute(operation, params)
        
        if success:
            new_df = operator.get_result()
            result = StepResult(
                step_num=step_num,
                description=description,
                operation=operation,
                status=StepStatus.SUCCESS,
                message=msg,
                rows_before=rows_before,
                rows_after=len(new_df),
            )
            results.append(result)
            return {
                **state,
                "df": new_df,
                "results": results,
                "current_step_idx": idx + 1,
                "error_message": "",
            }
        else:
            # Track retry count
            retry_count = 0
            for r in results:
                if r.step_num == step_num:
                    retry_count = r.retry_count + 1
                    results.remove(r)
                    break
            
            result = StepResult(
                step_num=step_num,
                description=description,
                operation=operation,
                status=StepStatus.FAILED,
                message=msg,
                rows_before=rows_before,
                retry_count=retry_count,
            )
            results.append(result)
            return {
                **state,
                "results": results,
                "error_message": msg,
            }
    except Exception as e:
        result = StepResult(
            step_num=step_num,
            description=description,
            operation=operation,
            status=StepStatus.FAILED,
            message=str(e),
            rows_before=rows_before,
        )
        results.append(result)
        return {
            **state,
            "results": results,
            "error_message": str(e),
        }


def validate_step_node(state: TransformationState) -> TransformationState:
    """Validate the result of the last step."""
    if not state["results"]:
        return state
    
    last_result = state["results"][-1]
    
    if last_result.status == StepStatus.SUCCESS:
        df = state["df"]
        
        if df is not None:
            # Warn if >90% of rows removed
            if last_result.rows_before > 0:
                removal_rate = 1 - (last_result.rows_after / last_result.rows_before)
                if removal_rate > 0.9:
                    last_result.message += f" ⚠️ Warning: Removed {removal_rate*100:.1f}% of rows"
            
            # Warn if dataset is now empty
            if len(df) == 0:
                last_result.message += " ⚠️ Warning: Dataset is now empty"
    
    return state


def finalize_node(state: TransformationState) -> TransformationState:
    """Create final summary message."""
    results = state["results"]
    df = state["df"]
    
    successful = sum(1 for r in results if r.status == StepStatus.SUCCESS)
    failed = sum(1 for r in results if r.status == StepStatus.FAILED)
    total = len(state["steps"])
    
    # Build summary
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
    
    if df is not None and len(df) > 0:
        parts.append(f"\n**Result:** {len(df)} rows × {len(df.columns)} columns")
    
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
        # Max retries reached - skip to next step
        state["current_step_idx"] = state.get("current_step_idx", 0) + 1
        if state["current_step_idx"] >= len(state["steps"]):
            return "finalize"
        return "continue"
    
    return "continue"


def needs_approval_check(state: TransformationState) -> Literal["wait_approval", "execute"]:
    """Check if we need approval."""
    if state["needs_approval"] and not state["approval_granted"]:
        return "wait_approval"
    return "execute"


# ============================================================================
# Build Transformation Graph
# ============================================================================

def create_transformation_graph() -> StateGraph:
    """Create the LangGraph workflow for dataset transformations."""
    
    graph = StateGraph(TransformationState)
    
    # Add nodes
    graph.add_node("plan", plan_node)
    graph.add_node("load_data", load_data_node)
    graph.add_node("check_approval", check_approval_node)
    graph.add_node("execute_step", execute_step_node)
    graph.add_node("validate", validate_step_node)
    graph.add_node("finalize", finalize_node)
    
    # Define edges
    graph.set_entry_point("plan")
    graph.add_edge("plan", "load_data")
    graph.add_edge("load_data", "check_approval")
    
    # Conditional: approval needed?
    graph.add_conditional_edges(
        "check_approval",
        needs_approval_check,
        {
            "execute": "execute_step",
            "wait_approval": "finalize",
        }
    )
    
    # After execution, validate
    graph.add_edge("execute_step", "validate")
    
    # After validation, decide: retry, continue, or finish
    graph.add_conditional_edges(
        "validate",
        should_retry_or_continue,
        {
            "retry": "execute_step",
            "continue": "check_approval",
            "finalize": "finalize",
        }
    )
    
    # End after finalize
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
) -> tuple[bool, str, Optional[pd.DataFrame]]:
    """Execute transformation workflow and return result."""
    
    graph = create_transformation_graph()
    
    initial_state: TransformationState = {
        "user_message": user_message,
        "data_path": data_path,
        "columns": columns,
        "steps": [],
        "current_step_idx": 0,
        "df": None,
        "results": [],
        "needs_approval": False,
        "approval_granted": True,
        "error_message": "",
        "max_retries": max_retries,
        "final_message": "",
        "success": False,
    }
    
    final_state = await graph.ainvoke(initial_state)
    
    return (
        final_state["success"],
        final_state["final_message"],
        final_state["df"],
    )


# ============================================================================
# Legacy Orchestrator Class (for backwards compatibility)
# ============================================================================

class Orchestrator:
    """Dataset orchestrator - wraps workflow functions for backwards compatibility."""
    
    def __init__(
        self,
        raw_storage: Path = RAW_STORAGE,
        curated_storage: Path = CURATED_STORAGE
    ) -> None:
        self.raw_storage = raw_storage
        self.curated_storage = curated_storage
        self.raw_storage.mkdir(parents=True, exist_ok=True)
        self.curated_storage.mkdir(parents=True, exist_ok=True)

    async def run_pipeline(
        self,
        session: AsyncSession,
        dataset_id: str,
        source_path: Path,
        approved_fixes: Optional[List[Dict]] = None,
    ) -> DatasetState:
        """Run upload pipeline."""
        return await process_upload(session, dataset_id, source_path)

    async def _get_state(self, session: AsyncSession, dataset_id: str) -> DatasetState:
        """Get dataset state."""
        return await get_dataset_state(session, dataset_id)

    async def _upsert_state(self, session: AsyncSession, state: DatasetState) -> DatasetState:
        """Update dataset state."""
        return await upsert_dataset_state(session, state)
