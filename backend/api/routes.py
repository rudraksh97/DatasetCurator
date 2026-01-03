from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from orchestrator.workflow import Orchestrator
from llm import classify_intent, chat_with_agent
from agents.data_ops import DataOperator

router = APIRouter()
orchestrator = Orchestrator()


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
    if intent == "show_data":
        if not has_data:
            response = "**No data loaded.**\n\n"
            response += "Please upload a CSV file using the attachment button."
        else:
            try:
                df = pd.read_csv(state.curated_path or state.raw_path)
                response = f"**Preview** ({len(df)} rows, {len(df.columns)} cols)\n\n"
                response += _df_to_markdown(df, max_rows=10)
                if len(df) > 10:
                    response += f"\n\n*Showing first 10 of {len(df)} rows.*"
            except Exception as e:
                response = f"Error: {str(e)}"
    
    elif intent == "transform_data":
        if not has_data:
            response = "**No data to transform.** Load a dataset first!"
        else:
            operation = params.pop("operation", None)
            if not operation:
                response = "I couldn't understand what transformation you want. Try:\n"
                response += "- *'remove column X'*\n- *'add column Y where Z > 10'*\n- *'drop rows where X = value'*"
            else:
                try:
                    df = pd.read_csv(state.curated_path or state.raw_path)
                    operator = DataOperator(df)
                    success, msg = operator.execute(operation, params)
                    
                    if success:
                        updated_df = operator.get_result()
                        version = state.current_version + 1
                        new_path = Path("storage/curated") / f"{state.dataset_id}_v{version}.csv"
                        updated_df.to_csv(new_path, index=False)
                        
                        state.curated_path = str(new_path)
                        state.current_version = version
                        state.transformation_log.append({
                            "operation": operation,
                            "params": params,
                            "timestamp": timestamp,
                        })
                        
                        explanation = intent_result.get("explanation", "Applied transformation")
                        response = f"**{explanation}**\n\n{msg}\n\n{operator.get_summary()}"
                        response += f"\n\n**Preview:**\n\n{_df_to_markdown(updated_df, 5)}"
                    else:
                        response = f"{msg}\n\n**Available columns:** {', '.join(columns)}"
                except Exception as e:
                    response = f"Error: {str(e)}"
    
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
