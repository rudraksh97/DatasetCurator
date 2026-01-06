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

from config import settings
from db import get_session, AsyncSessionLocal
from embeddings import embed_dataset
from orchestrator.workflow import (
    CURATED_STORAGE,
    execute_transformation,
    get_dataset_state,
    upsert_dataset_state,
)
from repositories.dataset import DatasetRepository
from services.llm import ChatService, IntentClassifierService

router = APIRouter()


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
    rows = [
        "| " + " | ".join(str(v) if pd.notna(v) else "*null*" for v in row) + " |" 
        for _, row in df_subset.iterrows()
    ]
    return "\n".join([headers, separator] + rows)


def _preview_df(path: Path, page: int = 1, page_size: int = 50) -> Dict[str, object]:
    """Get paginated preview of dataframe.
    
    Args:
        path: Path to CSV file.
        page: Page number (1-indexed).
        page_size: Number of rows per page.
    
    Returns:
        Dictionary containing paginated data and metadata.
    
    Raises:
        HTTPException: If the file cannot be read.
    """
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Data file not found: {path.name}")
    
    try:
        df = pd.read_csv(path)
        total_rows = len(df)
        total_pages = (total_rows + page_size - 1) // page_size
        
        skip = (page - 1) * page_size
        page_data = df.iloc[skip:skip + page_size].to_dict(orient="records")
        
        return {
            "data": page_data,
            "page": page,
            "page_size": page_size,
            "total_rows": total_rows,
            "total_pages": total_pages,
        }
    except pd.errors.EmptyDataError:
        return {
            "data": [],
            "page": 1,
            "page_size": page_size,
            "total_rows": 0,
            "total_pages": 0,
            "warning": "File is empty",
        }
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
    except Exception as e:
        print(f"[API] Error reading preview for {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read data file: {str(e)}")


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
    from orchestrator.workflow import process_upload
    
    target_dir = settings.storage.raw_path
    target_path = target_dir / f"{dataset_id}_{file.filename}"
    
    content = await file.read()
    target_path.write_bytes(content)
    
    if not target_path.exists():
        raise HTTPException(status_code=400, detail="Failed to save file")

    state = await process_upload(session, dataset_id, target_path)
    
    df = pd.read_csv(target_path)
    row_count = int(len(df))
    column_count = int(len(df.columns))

    preview_data = _preview_df(target_path, page=1, page_size=50)
    
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
    session: AsyncSession = Depends(get_session),
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
    repo = DatasetRepository(session)
    
    try:
        state = await repo.get(dataset_id)
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
    
    df_header = pd.read_csv(Path(data_path), nrows=1)
    column_count = len(df_header.columns)
    
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
    session: AsyncSession = Depends(get_session),
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
    repo = DatasetRepository(session)
    
    try:
        state = await repo.get(dataset_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not state.curated_path:
        raise HTTPException(status_code=404, detail="No processed file available")

    path = Path(state.curated_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(path, media_type="text/csv", filename=path.name)


class ChatHandler:
    """Handler for chat operations, encapsulating all chat-related logic."""
    
    def __init__(self, session: AsyncSession, dataset_id: str):
        """Initialize the chat handler.
        
        Args:
            session: Database session.
            dataset_id: Dataset identifier.
        """
        self._session = session
        self._dataset_id = dataset_id
        self._repo = DatasetRepository(session)
        self._intent_classifier = IntentClassifierService()
        self._chat_service = ChatService()
    
    async def get_context(self) -> tuple:
        """Get dataset state and context for chat.
        
        Returns:
            Tuple of (state, has_data, columns).
        """
        try:
            state = await self._repo.get(self._dataset_id)
            has_data = bool(state.curated_path or state.raw_path)
            columns = list(state.schema.keys()) if state.schema else []
            return state, has_data, columns
        except ValueError:
            return None, False, []
    
    async def handle_transform(
        self, 
        state, 
        user_msg: str, 
        has_data: bool, 
        columns: list,
    ) -> str:
        """Handle data transformation intent.
        
        Args:
            state: Dataset state.
            user_msg: User's message.
            has_data: Whether data is loaded.
            columns: Available columns.
        
        Returns:
            Response message.
        """
        if not has_data:
            return "**No data to transform.** Load a dataset first!"
        
        data_path = state.curated_path or state.raw_path
        success, final_message, final_df, is_analysis = await execute_transformation(
            user_message=user_msg,
            data_path=data_path,
            columns=columns,
            max_retries=1,
        )
        
        if final_df is not None and success:
            # Only save to curated storage if NOT analysis mode
            if not is_analysis:
                version = state.current_version + 1
                new_path = CURATED_STORAGE / f"{state.dataset_id}_v{version}.csv"
                final_df.to_csv(new_path, index=False)
                
                state.curated_path = str(new_path)
                state.current_version = version
            
            # Show more rows for small results, paginate for larger ones
            total_rows = len(final_df)
            if total_rows <= 20:
                # Small result - show all rows
                preview = _df_to_markdown(final_df, total_rows)
            elif total_rows <= 100:
                # Medium result - show first 20 with note
                preview = _df_to_markdown(final_df, 20)
                preview += f"\n\n*Showing 20 of {total_rows} rows*"
            else:
                # Large result - show first 10 with note
                preview = _df_to_markdown(final_df, 10)
                preview += f"\n\n*Showing 10 of {total_rows} rows. Check the Data Preview panel for full paginated view.*"
            
            return final_message + f"\n\n**Preview:**\n\n{preview}"
        
        return final_message if final_message else "I couldn't understand your transformation request."
    
    async def handle_chat(
        self, 
        state, 
        user_msg: str, 
        has_data: bool, 
        columns: list,
    ) -> str:
        """Handle general chat intent.
        
        Args:
            state: Dataset state.
            user_msg: User's message.
            has_data: Whether data is loaded.
            columns: Available columns.
        
        Returns:
            Response message.
        """
        if not has_data:
            return (
                "**Welcome!** I can help you clean and transform datasets.\n\n"
                "**Try:**\n"
                "- Upload a CSV using the attachment button\n"
                "- Transform your data with commands like *'remove column X'* or *'filter where Y > 10'*"
            )
        
        context = {"columns": columns}
        history = [{"role": m.get("role"), "content": m.get("content")} for m in state.chat_history]
        data_path = state.curated_path or state.raw_path
        
        return await self._chat_service.chat(
            user_msg, 
            data_path=data_path, 
            context=context, 
            history=history,
            session=self._session, 
            dataset_id=self._dataset_id,
        )
    
    async def save_history(
        self, 
        state, 
        user_msg: str, 
        response: str, 
        timestamp: str,
    ) -> None:
        """Save chat messages to history.
        
        Args:
            state: Dataset state.
            user_msg: User's message.
            response: Assistant's response.
            timestamp: Message timestamp.
        """
        if state:
            state.chat_history.append({"role": "user", "content": user_msg, "timestamp": timestamp})
            state.chat_history.append({"role": "assistant", "content": response, "timestamp": timestamp})
            await self._repo.save(state)


@router.post("/chat/{dataset_id}")
async def chat(
    dataset_id: str, 
    message: ChatMessageRequest, 
    session: AsyncSession = Depends(get_session),
) -> ChatResponse:
    """Chat with the dataset curator using LLM function calling.
    
    Args:
        dataset_id: Unique identifier for the dataset.
        message: Chat message from the user.
        session: Database session.
    
    Returns:
        ChatResponse with user and assistant messages.
    """
    handler = ChatHandler(session, dataset_id)
    user_msg = message.content
    timestamp = pd.Timestamp.utcnow().isoformat()
    
    state, has_data, columns = await handler.get_context()

    intent_classifier = IntentClassifierService()
    intent_result = await intent_classifier.classify(user_msg, has_data=has_data, columns=columns)
    intent = intent_result.get("intent", "chat")
    
    if intent_result.get("error"):
        print(f"[Chat] Intent classification warning: {intent_result.get('error')}")

    if intent in ("transform_data", "multi_transform"):
        response = await handler.handle_transform(state, user_msg, has_data, columns)
    else:
        response = await handler.handle_chat(state, user_msg, has_data, columns)

    await handler.save_history(state, user_msg, response, timestamp)

    return ChatResponse(user_message=user_msg, assistant_message=response)
