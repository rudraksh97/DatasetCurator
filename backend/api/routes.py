from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, List, AsyncGenerator

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from orchestrator.workflow import (
    Orchestrator,
    execute_transformation,
    execute_transformation_streaming,
    WorkflowEvent,
    CURATED_STORAGE,
)
from llm import classify_intent, chat_with_agent, create_execution_plan
from agents.data_ops import DataOperator

router = APIRouter()
orchestrator = Orchestrator()


def _sse_event(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event message."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


class ChatMessageRequest(BaseModel):
    content: str


class ChatResponse(BaseModel):
    user_message: str
    assistant_message: str


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 10) -> str:
    """Convert DataFrame to markdown table."""
    df_subset = df.head(max_rows)
    headers = "| " + " | ".join(str(c) for c in df_subset.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df_subset.columns) + " |"
    rows = ["| " + " | ".join(str(v) if pd.notna(v) else "*null*" for v in row) + " |" 
            for _, row in df_subset.iterrows()]
    return "\n".join([headers, separator] + rows)


def _preview_df(path: Path, page: int = 1, page_size: int = 50) -> Dict[str, object]:
    """Get paginated preview of dataframe."""
    try:
        df = pd.read_csv(path)
        total_rows = len(df)
        total_pages = (total_rows + page_size - 1) // page_size  # Ceiling division
        
        # Calculate skip and take
        skip = (page - 1) * page_size
        take = page_size
        
        # Get the page of data
        page_data = df.iloc[skip:skip + take].to_dict(orient="records")
        
        return {
            "data": page_data,
            "page": page,
            "page_size": page_size,
            "total_rows": total_rows,
            "total_pages": total_pages,
        }
    except Exception:
        return {
            "data": [],
            "page": 1,
            "page_size": page_size,
            "total_rows": 0,
            "total_pages": 0,
        }


@router.post("/upload")
async def upload_dataset(
    dataset_id: str = Form(...),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> Dict[str, object]:
    """Upload and process a dataset."""
    target_dir = Path("storage/raw")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{dataset_id}_{file.filename}"
    
    content = await file.read()
    target_path.write_bytes(content)
    
    if not target_path.exists():
        raise HTTPException(status_code=400, detail="Failed to save file")

    state = await orchestrator.run_pipeline(session, dataset_id, target_path)
    
    # Get actual row and column counts from the full file
    df = pd.read_csv(target_path)
    row_count = int(len(df))
    column_count = int(len(df.columns))

    preview_data = _preview_df(target_path, page=1, page_size=50)
    
    return {
        "dataset_id": dataset_id,
        "preview": preview_data["data"],
        "row_count": row_count,
        "column_count": column_count,
        "page": preview_data["page"],
        "page_size": preview_data["page_size"],
        "total_rows": preview_data["total_rows"],
        "total_pages": preview_data["total_pages"],
    }


@router.get("/preview/{dataset_id}")
async def get_preview(
    dataset_id: str, 
    page: int = 1,
    page_size: int = 50,
    session: AsyncSession = Depends(get_session)
) -> Dict[str, object]:
    """Get paginated data preview."""
    try:
        state = await orchestrator._get_state(session, dataset_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    data_path = state.curated_path or state.raw_path
    
    if not data_path:
        return {
            "dataset_id": dataset_id,
            "preview": [],
            "row_count": 0,
            "column_count": 0,
            "page": 1,
            "page_size": page_size,
            "total_rows": 0,
            "total_pages": 0,
        }
    
    preview_data = _preview_df(Path(data_path), page=page, page_size=page_size)
    
    # Get column count
    try:
        df = pd.read_csv(Path(data_path), nrows=1)
        column_count = len(df.columns)
    except Exception:
        column_count = 0
    
    return {
        "dataset_id": dataset_id,
        "preview": preview_data["data"],
        "row_count": preview_data["total_rows"],
        "column_count": column_count,
        "page": preview_data["page"],
        "page_size": preview_data["page_size"],
        "total_rows": preview_data["total_rows"],
        "total_pages": preview_data["total_pages"],
    }


@router.get("/download/{dataset_id}/file")
async def download_file(dataset_id: str, session: AsyncSession = Depends(get_session)):
    """Download the processed dataset file."""
    try:
        state = await orchestrator._get_state(session, dataset_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not state.curated_path:
        raise HTTPException(status_code=404, detail="No processed file available")

    path = Path(state.curated_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(path, media_type="text/csv", filename=path.name)


@router.post("/chat/{dataset_id}")
async def chat(
    dataset_id: str, message: ChatMessageRequest, session: AsyncSession = Depends(get_session)
) -> ChatResponse:
    """Chat with the dataset curator using LLM function calling."""
    user_msg = message.content
    timestamp = pd.Timestamp.utcnow().isoformat()
    
    # Get existing state
    try:
        state = await orchestrator._get_state(session, dataset_id)
        has_data = bool(state.curated_path or state.raw_path)
        columns = list(state.schema.keys()) if state.schema else []
    except ValueError:
        state = None
        has_data = False
        columns = []

    # Use LLM to classify intent
    intent_result = await classify_intent(
        user_msg,
        has_data=has_data,
        columns=columns,
    )
    intent = intent_result.get("intent", "chat")
    params = intent_result.get("params", {})

    response = ""

    # Handle intents
    if intent in ("transform_data", "multi_transform"):
        # All transformations go through LangGraph (handles both single & multi-step)
        if not has_data:
            response = "**No data to transform.** Load a dataset first!"
        else:
            data_path = state.curated_path or state.raw_path
            success, final_message, final_df = await execute_transformation(
                user_message=user_msg,
                data_path=data_path,
                columns=columns,
                max_retries=1,
            )
            
            if final_df is not None and success:
                version = state.current_version + 1
                new_path = CURATED_STORAGE / f"{state.dataset_id}_v{version}.csv"
                final_df.to_csv(new_path, index=False)
                
                state.curated_path = str(new_path)
                state.current_version = version
                
                response = final_message + f"\n\n**Preview:**\n\n{_df_to_markdown(final_df, 5)}"
            else:
                response = final_message if final_message else "I couldn't understand your transformation request."
    
    else:  # chat
        if not has_data:
            response = "**Welcome!** I can help you clean and transform datasets.\n\n"
            response += "**Try:**\n"
            response += "- Upload a CSV using the attachment button\n"
            response += "- Transform your data with commands like *'remove column X'* or *'filter where Y > 10'*"
        else:
            context = {"columns": columns}
            history = [{"role": m.get("role"), "content": m.get("content")} for m in state.chat_history[-6:]]
            data_path = state.curated_path or state.raw_path
            try:
                response = await chat_with_agent(user_msg, data_path=data_path, context=context, history=history)
            except Exception as e:
                response = f"Error: {str(e)}\n\nTry: *'show data'* or *'remove column X'*"

    # Save chat history
    if state:
        state.chat_history.append({"role": "user", "content": user_msg, "timestamp": timestamp})
        state.chat_history.append({"role": "assistant", "content": response, "timestamp": timestamp})
        await orchestrator._upsert_state(session, state)

    return ChatResponse(user_message=user_msg, assistant_message=response)


@router.post("/chat/{dataset_id}/stream")
async def chat_stream(
    dataset_id: str, message: ChatMessageRequest, session: AsyncSession = Depends(get_session)
):
    """Stream chat responses with real-time step updates for multi-step operations."""
    user_msg = message.content
    timestamp = pd.Timestamp.utcnow().isoformat()

    async def generate_stream() -> AsyncGenerator[str, None]:
        nonlocal session
        
        # Get existing state
        try:
            state = await orchestrator._get_state(session, dataset_id)
            has_data = bool(state.curated_path or state.raw_path)
            columns = list(state.schema.keys()) if state.schema else []
        except ValueError:
            state = None
            has_data = False
            columns = []

        # Classify intent
        intent_result = await classify_intent(user_msg, has_data=has_data, columns=columns)
        intent = intent_result.get("intent", "chat")
        params = intent_result.get("params", {})

        # Handle all transformations with unified LangGraph streaming workflow
        if intent in ("transform_data", "multi_transform") and has_data:
            data_path = state.curated_path or state.raw_path
            final_message = ""
            final_df = None
            executed_count = 0
            total_steps = 0
            
            # Use unified LangGraph workflow with streaming
            async for event in execute_transformation_streaming(
                user_message=user_msg,
                data_path=data_path,
                columns=columns,
                max_retries=1,  # Retry failed steps once
            ):
                # Convert WorkflowEvent to SSE
                if event.event_type == "plan":
                    total_steps = event.data.get("total_steps", 0)
                    yield _sse_event("plan", {
                        "total_steps": total_steps,
                        "steps": event.data.get("steps", []),
                    })
                
                elif event.event_type == "step_start":
                    yield _sse_event("step_start", {
                        "step": event.data.get("step"),
                        "description": event.data.get("description"),
                    })
                
                elif event.event_type == "step_complete":
                    # Include retry info if step was retried
                    msg = event.data.get("message", "")
                    if event.data.get("retried"):
                        msg += " (retried)"
                    
                    yield _sse_event("step_complete", {
                        "step": event.data.get("step"),
                        "success": event.data.get("success"),
                        "message": msg,
                        "rows_before": event.data.get("rows_before"),
                        "rows_after": event.data.get("rows_after"),
                    })
                
                elif event.event_type == "done":
                    final_message = event.data.get("final_message", "")
                    executed_count = event.data.get("total_executed", 0)
                    
                    # Reload the transformed data for saving
                    if event.data.get("success") and executed_count > 0:
                        # Re-execute to get final DataFrame (graph doesn't return it via streaming)
                        try:
                            df = pd.read_csv(data_path)
                            # Re-run the plan to get final state
                            plan = await create_execution_plan(user_msg, columns)
                            for step in plan.get("steps", []):
                                op = step.get("operation")
                                params = step.get("params", {})
                                if op:
                                    operator = DataOperator(df)
                                    success, _ = operator.execute(op, params)
                                    if success:
                                        df = operator.get_result()
                            final_df = df
                        except Exception:
                            final_df = None
            
            # Save final result
            if final_df is not None and executed_count > 0:
                version = state.current_version + 1
                new_path = CURATED_STORAGE / f"{state.dataset_id}_v{version}.csv"
                final_df.to_csv(new_path, index=False)
                
                state.curated_path = str(new_path)
                state.current_version = version
                
                # Add preview to final message
                final_message += f"\n\n**Preview:**\n\n{_df_to_markdown(final_df, 5)}"
            
            # Save chat history
            state.chat_history.append({"role": "user", "content": user_msg, "timestamp": timestamp})
            state.chat_history.append({"role": "assistant", "content": final_message, "timestamp": timestamp})
            await orchestrator._upsert_state(session, state)
            
            yield _sse_event("done", {
                "success": executed_count > 0,
                "total_executed": executed_count,
                "total_steps": total_steps,
                "final_message": final_message
            })

        else:
            # Chat intent
            if has_data:
                context = {"columns": columns}
                history = [{"role": m.get("role"), "content": m.get("content")} for m in state.chat_history[-6:]]
                data_path = state.curated_path or state.raw_path
                response = await chat_with_agent(user_msg, data_path=data_path, context=context, history=history)
            else:
                response = "**Welcome!** Upload a CSV to get started."

            # Save chat history for non-streaming
            if state:
                state.chat_history.append({"role": "user", "content": user_msg, "timestamp": timestamp})
                state.chat_history.append({"role": "assistant", "content": response, "timestamp": timestamp})
                await orchestrator._upsert_state(session, state)

            yield _sse_event("message", {"content": response})
            yield _sse_event("done", {"success": True, "final_message": response})

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
