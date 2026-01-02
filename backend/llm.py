"""OpenRouter LLM client for agent reasoning and chat."""
from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"


def get_client() -> AsyncOpenAI:
    """Get async OpenAI client configured for OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set in environment")
    return AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )


async def chat_completion(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """Send a chat completion request to OpenRouter."""
    client = get_client()
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


async def analyze_dataset(
    schema: Dict[str, Any],
    sample_data: List[Dict[str, Any]],
    quality_issues: List[Dict[str, Any]],
) -> str:
    """Ask the LLM to analyze a dataset and provide insights."""
    system_prompt = """You are an expert data analyst and dataset curator. 
Analyze the provided dataset schema, sample data, and quality issues.
Provide clear, actionable insights about:
1. Data quality assessment
2. Potential issues and their severity
3. Recommended fixes
4. Best practices for using this dataset

Be concise but thorough. Use markdown formatting."""

    user_content = f"""## Dataset Schema
```json
{schema}
```

## Sample Data (first rows)
```json
{sample_data[:5]}
```

## Detected Quality Issues
```json
{quality_issues}
```

Please analyze this dataset and provide your assessment."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return await chat_completion(messages, temperature=0.3)


async def generate_fix_suggestions(
    column: str,
    issue: str,
    sample_values: List[Any],
) -> str:
    """Ask the LLM to suggest fixes for a specific data quality issue."""
    system_prompt = """You are a data cleaning expert. Given a column, an issue description, 
and sample values, suggest the best approach to fix the issue.
Be specific and actionable. Provide code snippets if helpful."""

    user_content = f"""Column: {column}
Issue: {issue}
Sample values: {sample_values}

What's the best way to fix this issue?"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return await chat_completion(messages, temperature=0.3, max_tokens=1024)


async def chat_with_agent(
    user_message: str,
    context: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Have a conversation with the dataset curator agent."""
    system_prompt = """You are the Agentic Dataset Curator, an AI assistant that helps users 
clean, analyze, and prepare datasets for machine learning and analytics.

You can help users:
- Understand their data quality issues
- Suggest and apply fixes
- Generate dataset documentation
- Answer questions about data cleaning best practices

Be helpful, concise, and actionable. Use markdown formatting for clarity.
If you don't have enough context to answer, ask clarifying questions."""

    if context:
        system_prompt += f"""

Current context:
- Dataset ID: {context.get('dataset_id', 'Not set')}
- Columns: {context.get('columns', 'Unknown')}
- Row count: {context.get('row_count', 'Unknown')}
- Quality issues: {context.get('issue_count', 0)} detected"""

    messages = [{"role": "system", "content": system_prompt}]
    
    if history:
        messages.extend(history)
    
    messages.append({"role": "user", "content": user_message})
    
    return await chat_completion(messages, temperature=0.7)
