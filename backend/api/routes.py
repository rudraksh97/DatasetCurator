"""API routes for the Dataset Curator.

This module defines all REST API endpoints for:
- Dataset upload and processing
- Data preview with pagination
- File download
- Chat-based data transformations
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session, AsyncSessionLocal
from orchestrator.workflow import (
    Orchestrator,
    execute_transformation,
    CURATED_STORAGE,
)
from llm import classify_intent, chat_with_agent
from embeddings import embed_dataset

router = APIRouter()
orchestrator = Orchestrator()


class ChatMessageRequest(BaseModel):
    """Request body for chat messages."""
    content: str


class ChatResponse(BaseModel):
    """Response body for chat messages."""
    user_message: str
    assistant_message: str


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 10) -> str:
    """Convert DataFrame to markdown table format.
    
    Args:
        df: DataFrame to convert.
        max_rows: Maximum number of rows to include.
    
    Returns:
        Markdown-formatted table string.
    """
    df_subset = df.head(max_rows)
    headers = "| " + " | ".join(str(c) for c in df_subset.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df_subset.columns) + " |"
    rows = ["| " + " | ".join(str(v) if pd.notna(v) else "*null*" for v in row) + " |" 
            for _, row in df_subset.iterrows()]
    return "\n".join([headers, separator] + rows)


def _preview_df(path: Path, page: int = 1, page_size: int = 50) -> Dict[str, object]:
    """Get paginated preview of dataframe.
    
    Args:
        path: Path to CSV file.
        page: Page number (1-indexed).
        page_size: Number of rows per page.
    
    Returns:
        Dictionary containing paginated data and metadata.
    """
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


async def _embed_dataset_background(dataset_id: str, file_path: str) -> None:
    """Background task to embed dataset for semantic search.
    
    Args:
        dataset_id: Unique identifier for the dataset.
        file_path: Path to the CSV file.
    """
    try:
        async with AsyncSessionLocal() as session:
            df = pd.read_csv(file_path)
            count = await embed_dataset(session, dataset_id, df)
            print(f"[Upload] Embedded {count} rows for dataset {dataset_id}")
    except Exception as e:
        print(f"[Upload] Failed to embed dataset {dataset_id}: {e}")


@router.post("/upload")
async def upload_dataset(
    background_tasks: BackgroundTasks,
    dataset_id: str = Form(...),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> Dict[str, object]:
    """Upload and process a dataset.
    
    Args:
        background_tasks: FastAPI background task manager.
        dataset_id: Unique identifier for the dataset.
        file: CSV file to upload.
        session: Database session.
    
    Returns:
        Dictionary with dataset info, preview data, and pagination metadata.
    """
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
    
    # Embed dataset in background for semantic search
    background_tasks.add_task(_embed_dataset_background, dataset_id, str(target_path))
    
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
    """Get paginated data preview.
    
    Args:
        dataset_id: Unique identifier for the dataset.
        page: Page number (1-indexed).
        page_size: Number of rows per page.
        session: Database session.
    
    Returns:
        Dictionary with preview data and pagination metadata.
    
    Raises:
        HTTPException: If dataset is not found.
    """
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
async def download_file(
    dataset_id: str, 
    session: AsyncSession = Depends(get_session)
) -> FileResponse:
    """Download the processed dataset file.
    
    Args:
        dataset_id: Unique identifier for the dataset.
        session: Database session.
    
    Returns:
        FileResponse with the CSV file.
    
    Raises:
        HTTPException: If dataset or file is not found.
    """
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
    dataset_id: str, 
    message: ChatMessageRequest, 
    session: AsyncSession = Depends(get_session)
) -> ChatResponse:
    """Chat with the dataset curator using LLM function calling.
    
    Args:
        dataset_id: Unique identifier for the dataset.
        message: Chat message from the user.
        session: Database session.
    
    Returns:
        ChatResponse with user and assistant messages.
    """
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
            history = [{"role": m.get("role"), "content": m.get("content")} for m in state.chat_history]
            data_path = state.curated_path or state.raw_path
            try:
                response = await chat_with_agent(
                    user_msg, data_path=data_path, context=context, history=history,
                    session=session, dataset_id=dataset_id
                )
            except Exception as e:
                response = f"Error: {str(e)}\n\nTry: *'show data'* or *'remove column X'*"

    # Save chat history
    if state:
        state.chat_history.append({"role": "user", "content": user_msg, "timestamp": timestamp})
        state.chat_history.append({"role": "assistant", "content": response, "timestamp": timestamp})
        await orchestrator._upsert_state(session, state)

    return ChatResponse(user_message=user_msg, assistant_message=response)
