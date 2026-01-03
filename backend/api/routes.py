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
from agents.dataset_fetcher import list_available_datasets, fetch_dataset
from agents.dataset_search import (
    search_datasets,
    cache_search_results,
    get_cached_results,
    format_search_results,
    download_from_search_result,
)

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


def _preview_df(path: Path) -> List[Dict[str, object]]:
    try:
        return pd.read_csv(path, nrows=20).to_dict(orient="records")
    except Exception:
        return []


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

    return {
        "dataset_id": dataset_id,
        "quality_issues": state.quality_issues,
        "preview": _preview_df(target_path),
    }


@router.get("/preview/{dataset_id}")
async def get_preview(dataset_id: str, session: AsyncSession = Depends(get_session)) -> Dict[str, object]:
    """Get data preview and quality issues."""
    try:
        state = await orchestrator._get_state(session, dataset_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    data_path = state.curated_path or state.raw_path
    
    return {
        "dataset_id": dataset_id,
        "preview": _preview_df(Path(data_path)) if data_path else [],
        "quality_issues": state.quality_issues,
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

    # Check if user has pending search results
    cached_results = get_cached_results(dataset_id)
    has_search_results = cached_results is not None and len(cached_results) > 0

    # Use LLM to classify intent
    intent_result = await classify_intent(
        user_msg,
        has_data=has_data,
        columns=columns,
        has_search_results=has_search_results,
    )
    intent = intent_result.get("intent", "chat")
    params = intent_result.get("params", {})

    response = ""

    # Handle intents
    if intent == "search_datasets":
        query = params.get("query", user_msg)
        response = f"🔎 Searching for **{query}** datasets...\n\n"
        
        results = search_datasets(query)
        if results:
            cache_search_results(dataset_id, results)
            response += format_search_results(results)
        else:
            response += "❌ No datasets found. Try different keywords or use quick-access datasets:\n\n"
            response += list_available_datasets()

    elif intent == "select_result":
        # Ensure selection is an integer
        try:
            selection = int(params.get("selection", 0))
        except (ValueError, TypeError):
            selection = 0
        
        if not cached_results:
            response = "No search results to select from. Try searching first:\n- *'find weather data'*\n- *'search for stock prices'*"
        elif selection < 1 or selection > len(cached_results):
            response = f"Please select a number between 1 and {len(cached_results)}."
        else:
            result = cached_results[selection - 1]
            response = f"⏳ Downloading **{result['title']}**...\n\n"
            
            save_path = Path("storage/raw") / f"{dataset_id}_search.csv"
            download_result = await download_from_search_result(result, save_path)
            
            if download_result["success"]:
                state = await orchestrator.run_pipeline(session, dataset_id, save_path)
                response = f"✅ **Loaded: {download_result['name']}**\n\n"
                response += f"📊 {download_result['rows']} rows, {download_result['columns']} columns\n"
                response += f"Columns: `{', '.join(download_result['column_names'][:10])}`"
                if len(download_result['column_names']) > 10:
                    response += f" + {len(download_result['column_names']) - 10} more"
                response += "\n\nSay **'show data'** to preview, or select another result (1-8)."
                # Keep cache so user can try different results
            else:
                response = f"❌ {download_result['error']}\n\nTry selecting a different result or search again."

    elif intent == "list_datasets":
        response = list_available_datasets()
        response += "\n\n💡 *Or search for any dataset: 'find climate data', 'search for stocks'*"

    elif intent == "fetch_dataset":
        dataset_name = params.get("dataset_name", "").lower()
        if not dataset_name:
            response = "Which dataset would you like?\n\n" + list_available_datasets()
        else:
            save_path = Path("storage/raw") / f"{dataset_id}_{dataset_name}.csv"
            result = await fetch_dataset(dataset_name, save_path)
            
            if result["success"]:
                state = await orchestrator.run_pipeline(session, dataset_id, save_path)
                response = f"✅ **Fetched '{dataset_name}'!**\n\n"
                response += f"📊 {result['rows']} rows, {result['columns']} columns\n"
                response += f"Columns: `{', '.join(result['column_names'])}`\n\n"
                response += "Say **'show data'** to preview or **'download'** to get the file."
            else:
                response = f"❌ {result['error']}\n\n💡 Try searching: *'find {dataset_name} data'*"
    
    elif intent == "show_data":
        if not has_data:
            response = "📂 **No data loaded.**\n\n"
            response += "- 🔍 **Search**: *'find weather data'*\n"
            response += "- ⚡ **Quick**: *'fetch titanic'*\n"
            response += "- 📎 **Upload**: Use the attachment button"
        else:
            try:
                df = pd.read_csv(state.curated_path or state.raw_path)
                response = f"📊 **Preview** ({len(df)} rows, {len(df.columns)} cols)\n\n"
                response += _df_to_markdown(df, max_rows=10)
                if len(df) > 10:
                    response += f"\n\n*Showing first 10 of {len(df)} rows.*"
            except Exception as e:
                response = f"❌ Error: {str(e)}"
    
    elif intent == "transform_data":
        if not has_data:
            response = "📂 **No data to transform.** Load a dataset first!"
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
                        response = f"🔧 **{explanation}**\n\n{msg}\n\n{operator.get_summary()}"
                        response += f"\n\n**Preview:**\n\n{_df_to_markdown(updated_df, 5)}"
                    else:
                        response = f"⚠️ {msg}\n\n**Available columns:** {', '.join(columns)}"
                except Exception as e:
                    response = f"❌ Error: {str(e)}"
    
    else:  # chat
        if not has_data:
            response = "👋 **Welcome!** I can help you find and clean datasets.\n\n"
            response += "**Try:**\n"
            response += "- 🔍 *'find datasets about climate'*\n"
            response += "- ⚡ *'fetch titanic'* (quick access)\n"
            response += "- 📎 Upload a CSV using the attachment button"
        else:
            context = {"columns": columns, "issues": state.quality_issues[:3] if state.quality_issues else []}
            history = [{"role": m.get("role"), "content": m.get("content")} for m in state.chat_history[-6:]]
            try:
                response = await chat_with_agent(user_msg, context=context, history=history)
            except Exception as e:
                response = f"Error: {str(e)}\n\nTry: *'show data'* or *'remove column X'*"

    # Save chat history
    if state:
        state.chat_history.append({"role": "user", "content": user_msg, "timestamp": timestamp})
        state.chat_history.append({"role": "assistant", "content": response, "timestamp": timestamp})
        await orchestrator._upsert_state(session, state)

    return ChatResponse(user_message=user_msg, assistant_message=response)
