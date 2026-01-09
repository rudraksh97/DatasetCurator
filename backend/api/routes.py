"""API routes for the Dataset Curator.

This module defines all REST API endpoints for:
- Dataset upload and processing
- Data preview with pagination
- File download
- Chat-based data transformations
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import uuid
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from db import get_session, AsyncSessionLocal
from embeddings import embed_dataset, delete_dataset_embeddings
from orchestrator.workflow import (
    CURATED_STORAGE,
    execute_transformation,
    get_dataset_state,
    upsert_dataset_state,
)
from repositories.dataset import DatasetRepository
from services.llm import ChatService, IntentClassifierService
from services.llm.client import LLMRateLimitError, LLMAPIError, FREE_LLM_MODELS
from services.storage import get_storage, S3Storage

router = APIRouter()


class ChatMessageRequest(BaseModel):
    """Request body for chat messages."""
    content: str
    model: str | None = None
    approval_granted: bool | None = None


class ChatResponse(BaseModel):
    """Response body for chat messages."""
    user_message: str
    assistant_message: str
    result_preview: List[Dict[str, Any]] | None = None
    result_metadata: Dict[str, Any] | None = None


@router.get("/llm/models")
async def list_llm_models() -> Dict[str, object]:
    """List available LLM models (currently free OpenRouter models).
    
    Returns:
        Dictionary with model metadata and default model id.
    """
    default_model = next((m["id"] for m in FREE_LLM_MODELS if m.get("is_default")), settings.llm.default_model)
    
    return {
        "default_model": default_model,
        "models": FREE_LLM_MODELS,
    }

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


async def _preview_df(path: str, page: int = 1, page_size: int = 50) -> Dict[str, object]:
    """Get paginated preview of dataframe.
    
    Args:
        path: Path or S3 key to CSV file.
        page: Page number (1-indexed).
        page_size: Number of rows per page.
    
    Returns:
        Dictionary containing paginated data and metadata.
    
    Raises:
        HTTPException: If the file cannot be read.
    """
    storage = get_storage()
    # Ensure path is string for S3 compatibility
    path_str = str(path)
    
    if not await storage.exists(path_str):
        raise HTTPException(status_code=404, detail=f"Data file not found: {Path(path).name}")
    
    try:
        # Use storage abstraction to read CSV (handles both local and S3)
        df = await storage.read_csv(path_str)
        
        total_rows = len(df)
        total_pages = (total_rows + page_size - 1) // page_size
        
        skip = (page - 1) * page_size
        page_data = df.iloc[skip:skip + page_size].replace({float('nan'): None}).to_dict(orient="records")
        
        return {
            "data": page_data,
            "page": page,
            "page_size": page_size,
            "total_rows": total_rows,
            "total_pages": total_pages,
        }
    except Exception as e:
        print(f"[API] Error reading preview for {path}: {e}")
        # Return empty result safely for new empty files
        return {
            "data": [],
            "page": 1, 
            "page_size": page_size,
            "total_rows": 0,
            "total_pages": 0,
            "warning": f"Could not read file: {str(e)}"
        }


async def _embed_dataset_background(dataset_id: str, file_path: str) -> None:
    """Background task to embed dataset for semantic search.
    
    Args:
        dataset_id: Unique identifier for the dataset.
        file_path: Path to the CSV file.
    """
    try:
        async with AsyncSessionLocal() as session:
            storage = get_storage()
            df = await storage.read_csv(file_path)
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
    
    # Generate unique dataset ID to prevent collisions
    # Append short unique suffix (first 8 chars of uuid4)
    unique_suffix = str(uuid.uuid4())[:8]
    dataset_id = f"{dataset_id}_{unique_suffix}"
    
    storage = get_storage()
    
    # Generate storage path
    if settings.storage.is_s3:
        # S3: use prefix-based paths
        raw_path = f"{settings.storage.raw_prefix}/{dataset_id}_{file.filename}"
    else:
        # Local: use local path
        raw_path = f"raw/{dataset_id}_{file.filename}"
    
    content = await file.read()
    await storage.write_file(raw_path, content)
    
    # For processing, we need the full path or key
    if settings.storage.is_s3:
        target_path = raw_path  # S3 key
    else:
        target_path = settings.storage.raw_path / f"{dataset_id}_{file.filename}"

    state = await process_upload(session, dataset_id, target_path)
    
    # Use storage abstraction for reading
    # NOTE: LocalStorage expects relative path (raw_path), not full path (target_path)
    # target_path is constructed as storage/raw/... for Process/LangGraph, 
    # but storage.read_csv/preview does base_path + path.
    # To be safe and consistent with write_file, we use raw_path here.
    df = await storage.read_csv(raw_path)
    row_count = int(len(df))
    column_count = int(len(df.columns))

    preview_data = await _preview_df(raw_path, page=1, page_size=50)
    
    # Background task also uses storage abstraction eventually? 
    # Let's check _embed_dataset_background. It probably uses storage service too.
    background_tasks.add_task(_embed_dataset_background, dataset_id, raw_path)
    
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
            "warning": "No data path available"
        }
    
    # Use async preview
    preview_data = await _preview_df(data_path, page=page, page_size=page_size)
    
    # Get column count from storage
    storage = get_storage()
    try:
        # Read just the header
        content = await storage.read_file(str(data_path))
        import io
        df_header = pd.read_csv(io.BytesIO(content), nrows=0)
        column_count = len(df_header.columns)
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


@router.get("/versions/{dataset_id}")
async def list_versions(
    dataset_id: str,
    session: AsyncSession = Depends(get_session),
) -> Dict[str, object]:
    """List all available versions of a dataset.
    
    Args:
        dataset_id: Unique identifier for the dataset.
        session: Database session.
    
    Returns:
        Dictionary with list of versions and current version.
    
    Raises:
        HTTPException: If dataset is not found.
    """
    repo = DatasetRepository(session)
    
    try:
        state = await repo.get(dataset_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Find all versioned files for this dataset
    versions = []
    for file_path in CURATED_STORAGE.glob(f"{dataset_id}_v*.csv"):
        # Extract version number from filename
        filename = file_path.stem  # e.g., "exam_score_prediction_v2"
        version_part = filename.split("_v")[-1]
        try:
            version_num = int(version_part)
            file_stat = file_path.stat()
            versions.append({
                "version": version_num,
                "filename": file_path.name,
                "size_bytes": file_stat.st_size,
                "modified_at": pd.Timestamp.fromtimestamp(file_stat.st_mtime).isoformat(),
            })
        except ValueError:
            continue
    
    # Sort by version number
    versions.sort(key=lambda v: v["version"])
    
    return {
        "dataset_id": dataset_id,
        "current_version": state.current_version,
        "versions": versions,
    }


@router.get("/download/{dataset_id}/file")
async def download_file(
    dataset_id: str,
    version: int | None = None,
    session: AsyncSession = Depends(get_session),
):
    """Download the processed dataset file.
    
    Args:
        dataset_id: Unique identifier for the dataset.
        version: Optional version number. If not provided, downloads the latest.
        session: Database session.
    
    Returns:
        FileResponse for local storage, or RedirectResponse for S3 presigned URL.
    
    Raises:
        HTTPException: If dataset or file is not found.
    """
    from fastapi.responses import RedirectResponse, StreamingResponse
    import io
    
    repo = DatasetRepository(session)
    storage = get_storage()
    
    try:
        state = await repo.get(dataset_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if version is not None:
        # Download specific version
        if settings.storage.is_s3:
            file_key = f"{settings.storage.curated_prefix}/{dataset_id}_v{version}.csv"
        else:
            file_key = f"curated/{dataset_id}_v{version}.csv"
        filename = f"{dataset_id}_v{version}.csv"
    else:
        # Download latest version
        if not state.curated_path:
            raise HTTPException(status_code=404, detail="No processed file available")
        file_key = str(state.curated_path)
        filename = Path(file_key).name if "/" in file_key else file_key

    # Check if file exists
    if not await storage.exists(file_key):
        raise HTTPException(
            status_code=404, 
            detail=f"File not found. Use /versions/{dataset_id} to see available versions."
        )
    
    # Handle S3 storage - use presigned URL redirect
    if settings.storage.is_s3 and isinstance(storage, S3Storage):
        presigned_url = storage.generate_presigned_url(file_key, expires_in=300)
        return RedirectResponse(url=presigned_url, status_code=307)
    
    # Handle local storage - use FileResponse
    local_path = storage.get_local_path(file_key)
    if local_path and local_path.exists():
        return FileResponse(local_path, media_type="text/csv", filename=filename)
    
    # Fallback: stream from storage
    content = await storage.read_file(file_key)
    return StreamingResponse(
        io.BytesIO(content),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


class ChatHandler:
    """Handler for chat operations, encapsulating all chat-related logic."""
    
    def __init__(self, session: AsyncSession, dataset_id: str, model: str | None = None):
        """Initialize the chat handler.
        
        Args:
            session: Database session.
            dataset_id: Dataset identifier.
        """
        self._session = session
        self._dataset_id = dataset_id
        self._repo = DatasetRepository(session)
        self._intent_classifier = IntentClassifierService(model=model)
        self._chat_service = ChatService(model=model)
    
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
        approval_granted: bool | None = None,
    ) -> tuple[str, List[Dict[str, Any]] | None, Dict[str, Any] | None]:
        """Handle data transformation intent.
        
        Args:
            state: Dataset state.
            user_msg: User's message.
            has_data: Whether data is loaded.
            columns: Available columns.
        
        Returns:
            Tuple of (Response message, Preview data, Metadata).
        """
        if not has_data:
            return "**No data to transform.** Load a dataset first!", None, None
        
        data_path = state.curated_path or state.raw_path
        success, final_message, final_df, is_analysis = await execute_transformation(
            user_message=user_msg,
            data_path=data_path,
            columns=columns,
            max_retries=1,
            approval_granted=bool(approval_granted),
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
            
            # Generate markdown preview for the message
            if total_rows <= 20:
                preview_md = _df_to_markdown(final_df, total_rows)
            elif total_rows <= 100:
                preview_md = _df_to_markdown(final_df, 20)
                preview_md += f"\n\n*Showing 20 of {total_rows} rows*"
            else:
            # Generate preview textual summary if needed, but DO NOT append markdown table
            # The UI now handles preview in a separate panel.
            # final_message += f"\n\n**Preview:**\n\n{preview_md}"
            
            # Create structured preview data for the UI
            preview_data = final_df.head(50).replace({float('nan'): None}).to_dict(orient="records")
            metadata = {
                "row_count": total_rows,
                "column_count": len(final_df.columns),
                "is_analysis": is_analysis
            }
            
            return final_message, preview_data, metadata
        
        return final_message if final_message else "I couldn't understand your transformation request.", None, None
    
    async def handle_chat(
        self, 
        state, 
        user_msg: str, 
        has_data: bool, 
        columns: list,
    ) -> tuple[str, None, None]:
        """Handle general chat intent.
        
        Args:
            state: Dataset state.
            user_msg: User's message.
            has_data: Whether data is loaded.
            columns: Available columns.
        
        Returns:
            Tuple of (Response message, None, None).
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
        
        response = await self._chat_service.chat(
            user_msg, 
            data_path=data_path, 
            context=context, 
            history=history,
            session=self._session, 
            dataset_id=self._dataset_id,
        )
        return response, None, None
    
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
    handler = ChatHandler(session, dataset_id, model=message.model)
    user_msg = message.content
    timestamp = pd.Timestamp.utcnow().isoformat()
    
    try:
        state, has_data, columns = await handler.get_context()

        # Use the handler's configured intent classifier (respects selected model)
        intent_result = await handler._intent_classifier.classify(user_msg, has_data=has_data, columns=columns)
        intent = intent_result.get("intent", "chat")
        
        if intent_result.get("error"):
            print(f"[Chat] Intent classification warning: {intent_result.get('error')}")

        if intent in ("transform_data", "multi_transform"):
            response, result_preview, result_metadata = await handler.handle_transform(
                state,
                user_msg,
                has_data,
                columns,
                approval_granted=message.approval_granted,
            )
        else:
            response, result_preview, result_metadata = await handler.handle_chat(state, user_msg, has_data, columns)

        await handler.save_history(state, user_msg, response, timestamp)

        return ChatResponse(
            user_message=user_msg, 
            assistant_message=response,
            result_preview=result_preview,
            result_metadata=result_metadata
        )
    
    except LLMRateLimitError as e:
        # Return rate limit error as a friendly message instead of 500
        response = str(e)
        if state:
            await handler.save_history(state, user_msg, response, timestamp)
        return ChatResponse(user_message=user_msg, assistant_message=response)
    
    except LLMAPIError as e:
        # Return API errors as friendly messages
        response = f"⚠️ **AI Service Error:** {str(e)}"
        if state:
            await handler.save_history(state, user_msg, response, timestamp)
        return ChatResponse(user_message=user_msg, assistant_message=response)


@router.delete("/dataset/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    session: AsyncSession = Depends(get_session),
) -> Dict[str, str]:
    """Delete a dataset and all associated resources.
    
    Args:
        dataset_id: Unique identifier for the dataset.
        session: Database session.
    
    Returns:
        Confirmation message.
    
    Raises:
        HTTPException: If dataset is not found.
    """
    repo = DatasetRepository(session)
    storage = get_storage()
    
    # 1. Get dataset state to find file paths
    try:
        state = await repo.get(dataset_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # 2. Delete main files from storage
    files_to_delete = []
    if state.raw_path:
        files_to_delete.append(state.raw_path)
    if state.curated_path:
        files_to_delete.append(state.curated_path)
        
    # 3. Find and delete versioned files
    # Note: list_files returns keys/paths relative to prefix if implemented that way,
    # but delete_file expects what list_files returns usually.
    # However, list_files(prefix) returns relative paths in our implementation?
    # S3Storage.list_files returns keys relative to prefix.
    # LocalStorage.list_files returns relative paths.
    
    # We should search for files starting with dataset_id in raw/ and curated/
    # But currently S3Storage prefix is global for the bucket in my implementation logic?
    # Let's check config.
    
    # S3Storage implementation:
    # def list_files(self, prefix: str) -> List[str]:
    #    ... returns filtered list ...
    
    # Simpler approach: Try to delete likely version files v1 to v20
    # or rely on specific naming convention if possible.
    
    # Better: list files in curated prefix and find matches
    if settings.storage.is_s3:
        curated_prefix = settings.storage.curated_prefix
        try:
            curated_files = await storage.list_files(curated_prefix)
            for file in curated_files:
                if f"/{dataset_id}_v" in f"/{file}" or file.startswith(f"{dataset_id}_v"):
                     # Reconstruct full path for deletion if needed?
                     # storage.list_files returns keys (minus global prefix).
                     # storage.delete_file expects same key format?
                     # Yes, both use _get_key internally.
                     files_to_delete.append(file)
        except Exception as e:
            print(f"[Delete] Error listing files for cleanup: {e}")
    else:
        # Local storage cleanup
        # storage/curated is usually where versions are
        # But storage.list_files might be tricky with directories.
        # Let's iterate versions based on current_version in state to be safe/efficient
        for v in range(1, state.current_version + 1):
             files_to_delete.append(f"curated/{dataset_id}_v{v}.csv")

    # Execute file deletions
    for file_path in files_to_delete:
        try:
            await storage.delete_file(str(file_path))
        except Exception as e:
            print(f"[Delete] Failed to delete file {file_path}: {e}")

    # 4. Delete embeddings
    await delete_dataset_embeddings(session, dataset_id)
    
    # 5. Delete dataset record (cascades to chat history if stored in JSON or relations)
    # DatasetState stores chat_history in JSON column, so deleting record deletes history.
    await repo.delete(dataset_id)
    
    return {"message": f"Dataset {dataset_id} and all associated data deleted successfully"}
