from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from orchestrator.workflow import Orchestrator
from llm import chat_with_agent, analyze_dataset

router = APIRouter()
orchestrator = Orchestrator()


class ApproveRequest(BaseModel):
    dataset_id: str
    fixes: List[Dict]


class ChatMessageRequest(BaseModel):
    content: str


class ChatResponse(BaseModel):
    user_message: str
    assistant_message: str
    history: List[Dict]


@router.get("/chat/{dataset_id}")
async def get_chat_history(
    dataset_id: str, session: AsyncSession = Depends(get_session)
) -> List[Dict[str, object]]:
    try:
        state = await orchestrator._get_state(session, dataset_id)  # pylint: disable=protected-access
        return state.chat_history
    except ValueError:
        return []


@router.post("/chat/{dataset_id}")
async def chat_with_curator(
    dataset_id: str, message: ChatMessageRequest, session: AsyncSession = Depends(get_session)
) -> ChatResponse:
    """Send a message to the curator agent and get an LLM-powered response."""
    try:
        state = await orchestrator._get_state(session, dataset_id)  # pylint: disable=protected-access
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload a dataset first.")

    # Build context from current state
    context = {
        "dataset_id": dataset_id,
        "columns": list(state.schema.keys()) if state.schema else [],
        "row_count": state.schema.get("row_count", "Unknown") if state.schema else "Unknown",
        "issue_count": len(state.quality_issues),
    }

    # Convert history to OpenAI format
    history = []
    for msg in state.chat_history[-10:]:  # Last 10 messages for context
        history.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

    # Get LLM response
    try:
        assistant_response = await chat_with_agent(message.content, context=context, history=history)
    except Exception as e:
        assistant_response = f"I encountered an error processing your request: {str(e)}"

    # Save both messages to history
    timestamp = pd.Timestamp.utcnow().isoformat()
    state.chat_history.append({"role": "user", "content": message.content, "timestamp": timestamp})
    state.chat_history.append({"role": "assistant", "content": assistant_response, "timestamp": timestamp})

    await orchestrator._upsert_state(session, state)  # pylint: disable=protected-access

    return ChatResponse(
        user_message=message.content,
        assistant_message=assistant_response,
        history=state.chat_history,
    )


@router.post("/analyze/{dataset_id}")
async def analyze_dataset_endpoint(
    dataset_id: str, session: AsyncSession = Depends(get_session)
) -> Dict[str, object]:
    """Get an LLM-powered analysis of the dataset."""
    try:
        state = await orchestrator._get_state(session, dataset_id)  # pylint: disable=protected-access
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Get sample data from raw file
    sample_data = []
    if state.raw_path:
        try:
            df = pd.read_csv(state.raw_path, nrows=10)
            sample_data = df.to_dict(orient="records")
        except Exception:
            pass

    try:
        analysis = await analyze_dataset(
            schema=state.schema or {},
            sample_data=sample_data,
            quality_issues=state.quality_issues,
        )
    except Exception as e:
        analysis = f"Error generating analysis: {str(e)}"

    return {"dataset_id": dataset_id, "analysis": analysis}


@router.post("/upload")
async def upload_dataset(
    dataset_id: str = Form(...),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> Dict[str, object]:
    target_dir = Path("storage/raw")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{dataset_id}_{file.filename}"
    content = await file.read()
    target_path.write_bytes(content)
    if not target_path.exists():
        raise HTTPException(status_code=400, detail="Failed to persist uploaded file")

    state = await orchestrator.run_pipeline(session, dataset_id, target_path, approved_fixes=[])
    preview = _preview_df(target_path)
    return {
        "dataset_id": dataset_id,
        "quality_issues": state.quality_issues,
        "curated_path": str(state.curated_path) if state.curated_path else None,
        "preview": preview,
    }


@router.get("/health/{dataset_id}")
async def get_health(dataset_id: str, session: AsyncSession = Depends(get_session)) -> Dict[str, object]:
    try:
        issues = await orchestrator.health_report(session, dataset_id)
    except ValueError as exc:  # pragma: no cover - lightweight guard
        raise HTTPException(status_code=404, detail=str(exc))
    return {"dataset_id": dataset_id, "issues": issues}


@router.post("/approve")
async def approve_fixes(
    payload: ApproveRequest, session: AsyncSession = Depends(get_session)
) -> Dict[str, object]:
    try:
        state = await orchestrator.apply_approved_fixes(session, payload.dataset_id, payload.fixes)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {
        "dataset_id": payload.dataset_id,
        "curated_path": str(state.curated_path) if state.curated_path else None,
        "version": state.current_version,
    }


@router.get("/download/{dataset_id}")
async def download_curated(
    dataset_id: str, session: AsyncSession = Depends(get_session)
) -> Dict[str, object]:
    try:
        state = await orchestrator._get_state(session, dataset_id)  # pylint: disable=protected-access
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {
        "dataset_id": dataset_id,
        "curated_path": str(state.curated_path) if state.curated_path else None,
    }


@router.get("/card/{dataset_id}")
async def get_card(dataset_id: str, session: AsyncSession = Depends(get_session)) -> Dict[str, object]:
    try:
        card = await orchestrator.get_dataset_card(session, dataset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"dataset_id": dataset_id, "dataset_card": card}


@router.get("/download/{dataset_id}/file")
async def download_curated_file(dataset_id: str, session: AsyncSession = Depends(get_session)):
    try:
        state = await orchestrator._get_state(session, dataset_id)  # pylint: disable=protected-access
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not state.curated_path:
        raise HTTPException(status_code=404, detail="Curated dataset not available")

    path = Path(state.curated_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Curated file missing on disk")

    return FileResponse(
        path,
        media_type="text/csv",
        filename=path.name,
    )


def _preview_df(path: Path) -> List[Dict[str, object]]:
    try:
        df = pd.read_csv(path, nrows=20)
        return df.to_dict(orient="records")
    except Exception:
        return []
