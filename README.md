# Dataset Curator

A conversational data transformation engine that translates natural language into structured DataFrame operations. Built on FastAPI, LangGraph, and Next.js.

## Overview

Dataset Curator enables non-technical users to clean and transform CSV datasets through natural language. The system decomposes user intents into atomic, retryable operations executed via a directed acyclic graph (DAG) workflow.

**Core capabilities:**
- Intent classification → transformation planning → multi-step execution
- Automatic retry with step-level isolation
- Immutable version history per transformation
- Semantic search over dataset content (pgvector)
- Function-calling chat for exploratory data queries

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Client Layer                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│  Next.js 14 (App Router)                                                     │
│  ├─ Chat interface                                                           │
│  ├─ Paginated data preview                                                   │
│  └─ Session/version management                                               │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │ HTTP/REST
┌────────────────────────────────────▼─────────────────────────────────────────┐
│                            API Layer (FastAPI)                               │
├──────────────────────────────────────────────────────────────────────────────┤
│  routes.py                                                                   │
│  ├─ POST /upload          → process_upload()                                 │
│  ├─ POST /chat/{id}       → intent_classifier → [transform | chat]           │
│  ├─ GET  /preview/{id}    → paginated DataFrame read                         │
│  └─ GET  /download/{id}   → file stream                                      │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────────┐
│   LLM Services      │  │   Workflow Engine   │  │   Data Layer            │
├─────────────────────┤  ├─────────────────────┤  ├─────────────────────────┤
│ • IntentClassifier  │  │ LangGraph DAG       │  │ • DatasetRepository     │
│ • Planner           │  │ ├─ plan             │  │ • DataOperator          │
│ • ChatService       │  │ ├─ load_data        │  │ • QueryRegistry         │
│ • OpenRouterClient  │  │ ├─ check_approval   │  │                         │
│                     │  │ ├─ execute_step     │  │ Storage:                │
│ Model:              │  │ ├─ validate         │  │ • PostgreSQL + pgvector │
│ Llama 3.3 70B       │  │ └─ finalize         │  │ • File system (CSV)     │
│ via OpenRouter      │  │                     │  │                         │
└─────────────────────┘  └─────────────────────┘  └─────────────────────────┘
```

---

## Request Lifecycle

### 1. Intent Classification

All `/chat` requests pass through a two-stage classifier:

```
User Message
     │
     ▼
┌─────────────────────────────┐
│    Intent Classifier        │
│    (transform | query)      │
└──────────────┬──────────────┘
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
   TRANSFORM         QUERY
       │               │
       ▼               ▼
   Planner        ChatService
   (LangGraph)    (function calling)
```

**Transform path:** User wants to mutate data ("remove nulls", "sort by date")
**Query path:** User wants to inspect or analyze ("what's the average?", "show duplicates")

### 2. Transformation Workflow (LangGraph)

Multi-step transformations execute through a state machine with retry semantics:

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                         ▼
                    ┌─────────┐
                    │  PLAN   │ ← LLM generates step list
                    └────┬────┘
                         │
                         ▼
                  ┌────────────┐
                  │ LOAD DATA  │
                  └─────┬──────┘
                        │
         ┌──────────────┴──────────────┐
         │                             │
         ▼                             ▼
  ┌─────────────────┐          ┌─────────────────┐
  │ CHECK APPROVAL  │──skip───▶│    FINALIZE     │
  └────────┬────────┘          └────────┬────────┘
           │                            │
           ▼                            ▼
  ┌─────────────────┐                 [END]
  │  EXECUTE STEP   │◀───────┐
  └────────┬────────┘        │
           │                 │
           ▼                 │
  ┌─────────────────┐        │
  │    VALIDATE     │        │
  └────────┬────────┘        │
           │                 │
    ┌──────┴──────┐          │
    │      │      │          │
 success  fail  max_retries  │
    │      │      │          │
    │      │      └──────────┼───▶ FINALIZE
    │      │                 │
    │      └─────retry───────┘
    │
    └───────────▶ CHECK APPROVAL (next step)
```

**Key invariants:**
- Each step is atomic—failure doesn't corrupt prior steps
- Retries use identical parameters (deterministic)
- Validation warns on >90% row removal or empty result
- Final state persists to PostgreSQL on completion

### 3. Chat/Query Flow

Data questions route to `ChatService` with OpenAI-compatible function calling:

```
User: "What's the average price by category?"
                    │
                    ▼
         ┌────────────────────┐
         │    ChatService     │
         │  (tools enabled)   │
         └─────────┬──────────┘
                   │
                   ▼
         ┌────────────────────┐
         │  LLM selects tool  │
         │  → get_statistics  │
         └─────────┬──────────┘
                   │
                   ▼
         ┌────────────────────┐
         │  QueryRegistry     │
         │  executes on DF    │
         └─────────┬──────────┘
                   │
                   ▼
         ┌────────────────────┐
         │  LLM synthesizes   │
         │  natural response  │
         └────────────────────┘
```

**Available query tools:** `find_columns`, `search_rows`, `get_row`, `get_value`, `get_random_value`, `calculate_ratio`, `get_statistics`, `group_by`, `list_columns`, `get_row_count`

---

## Data Operations

All transformations map to canonical primitives:

| Category | Operation | Primitive | Description |
|----------|-----------|-----------|-------------|
| **Row Filtering** | `filter_rows` | `filter_rows` | Keep rows matching predicate |
| | `drop_rows` | `filter_rows` | Remove rows matching predicate |
| | `drop_nulls` | `filter_rows` | Remove rows with NULL in column(s) |
| **Row Transform** | `fill_nulls` | `map_rows` | Replace NULL with value/method |
| | `lowercase` | `map_rows` | Convert column to lowercase |
| | `uppercase` | `map_rows` | Convert column to uppercase |
| | `strip_whitespace` | `map_rows` | Trim leading/trailing whitespace |
| | `replace_values` | `map_rows` | Find and replace values |
| **Column Ops** | `drop_column` | `drop_columns` | Remove single column |
| | `keep_columns` | `select_columns` | Keep only specified columns |
| | `rename_column` | `rename_columns` | Rename column |
| | `add_column` | `add_column` | Add static/computed column |
| | `add_conditional_column` | `add_column + map_rows` | Add column with conditional logic |
| | `convert_type` | `cast_column_types` | Change column dtype |
| **Dataset Ops** | `drop_duplicates` | `deduplicate_rows` | Remove duplicate rows |
| | `sort` | `sort_rows` | Sort by column |
| | `limit_rows` | `limit_rows` | Keep first/last N rows |
| | `sample_rows` | `sample_rows` | Random sample |
| **Aggregation** | `group_aggregate` | `group_rows + aggregate` | Group by with aggregations |
| **Quality** | `validate_schema` | `validate_schema` | Check types, nulls, uniqueness |
| | `quarantine_rows` | `quarantine_rows` | Separate invalid rows |

---

## Project Structure

```
dataset-curator/
├── backend/
│   ├── main.py                     # FastAPI entrypoint
│   ├── config.py                   # Pydantic settings (env-based)
│   ├── db.py                       # SQLAlchemy async engine + pgvector init
│   │
│   ├── api/
│   │   └── routes.py               # REST endpoints
│   │
│   ├── services/
│   │   ├── llm/
│   │   │   ├── client.py           # OpenRouter API client
│   │   │   ├── intent.py           # Intent classifier
│   │   │   ├── planner.py          # Multi-step plan generator
│   │   │   ├── chat.py             # Query/conversation handler
│   │   │   ├── tools.py            # Function calling tool definitions
│   │   │   └── prompts.py          # System prompt templates
│   │   │
│   │   └── queries/
│   │       ├── base.py             # QueryHandler protocol
│   │       ├── handlers.py         # Concrete implementations
│   │       └── registry.py         # Handler dispatch
│   │
│   ├── orchestrator/
│   │   └── workflow.py             # LangGraph state machine
│   │
│   ├── agents/
│   │   └── data_ops.py             # DataOperator (pandas transforms)
│   │
│   ├── models/
│   │   ├── dataset_state.py        # Pydantic domain model
│   │   └── db_models.py            # SQLAlchemy ORM models
│   │
│   ├── repositories/
│   │   └── dataset.py              # Data access layer
│   │
│   ├── protocols.py                # Shared type protocols
│   ├── embeddings.py               # Semantic search (pgvector)
│   └── data_loader.py              # Smart CSV loading (chunked/sampled)
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx                # Main UI
│   │   ├── layout.tsx              # App shell
│   │   └── globals.css             # Styles
│   │
│   ├── components/
│   │   ├── sidebar.tsx             # Session navigator
│   │   └── ui/                     # Reusable primitives
│   │
│   ├── lib/
│   │   └── api.ts                  # Typed API client
│   │
│   └── types/
│       └── api.ts                  # Shared types
│
├── storage/
│   ├── raw/                        # Uploaded originals
│   └── curated/                    # Versioned outputs
│
└── docker-compose.yml
```

---

## Configuration

All settings are environment-variable driven via Pydantic Settings.

### Required

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL async connection string (`postgresql+asyncpg://...`) |
| `OPENROUTER_API_KEY` | API key from [OpenRouter](https://openrouter.ai/keys) |

### LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_LLM_MODEL` | `meta-llama/llama-3.3-70b-instruct:free` | Model identifier |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | API base URL |

### Embedding / Semantic Search

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `EMBEDDING_DIM` | `384` | Vector dimension (must match model) |

### Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `RAW_STORAGE_PATH` | `storage/raw` | Upload directory |
| `CURATED_STORAGE_PATH` | `storage/curated` | Output directory |

### Data Processing

| Variable | Default | Description |
|----------|---------|-------------|
| `LARGE_FILE_SIZE_MB` | `100` | Threshold for "large file" handling |
| `LARGE_ROW_COUNT` | `1000000` | Threshold for row-based sampling |
| `DATA_SAMPLE_SIZE` | `10000` | Sample size for large file queries |
| `DATA_CHUNK_SIZE` | `50000` | Chunk size for streaming ops |

### Workflow

| Variable | Default | Description |
|----------|---------|-------------|
| `REQUIRE_APPROVAL_FOR_DESTRUCTIVE_OPS` | `false` | Gate destructive ops |
| `APPROVAL_ROW_THRESHOLD` | `1000` | Row count trigger for approval |

### CORS

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated origins |

---

## Deployment

### Docker Compose (Recommended)

```bash
# Clone repository
git clone <repository-url>
cd dataset-curator

# Configure environment
cat > .env << 'EOF'
OPENROUTER_API_KEY=sk-or-...
DATABASE_URL=postgresql+asyncpg://dataset_curator:dataset_curator@db:5432/dataset_curator
EOF

# Start services
docker compose up --build -d

# Verify
curl http://localhost:8000/healthcheck
```

**Service endpoints:**
- Frontend: `http://localhost:3000`
- API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

### Local Development

**Backend:**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Requires running PostgreSQL with pgvector
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/dataset_curator"
export OPENROUTER_API_KEY="sk-or-..."

uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**Tests:**
```bash
cd backend
pytest -v
```

---

## API Reference

### Upload Dataset

```http
POST /upload
Content-Type: multipart/form-data

dataset_id: <unique-identifier>
file: <csv-file>
```

**Response:**
```json
{
  "dataset_id": "sales_abc123",
  "preview": [...],
  "row_count": 10000,
  "column_count": 8,
  "page": 1,
  "page_size": 50,
  "total_rows": 10000,
  "total_pages": 200
}
```

### Chat / Transform

```http
POST /chat/{dataset_id}
Content-Type: application/json

{
  "content": "remove rows where revenue is null and sort by date descending"
}
```

**Response:**
```json
{
  "user_message": "remove rows where revenue is null and sort by date descending",
  "assistant_message": "**Executed 2 steps:**\n\nStep 1: Remove null revenue\n  ✓ Removed 45 rows..."
}
```

### Preview Data

```http
GET /preview/{dataset_id}?page=1&page_size=50
```

**Response:**
```json
{
  "dataset_id": "sales_abc123",
  "preview": [...],
  "row_count": 9955,
  "column_count": 8,
  "page": 1,
  "page_size": 50,
  "total_rows": 9955,
  "total_pages": 200
}
```

### Download

```http
GET /download/{dataset_id}/file
```

Returns CSV file stream with `Content-Disposition` header.

---

## Extending the System

### Adding a New Operation

1. **Add handler to `DataOperator`** (`agents/data_ops.py`):

```python
def _my_operation(self, params: Dict) -> str:
    """Base primitive: description."""
    # Implementation
    return "Result message"
```

2. **Register in operation map** (same file, `execute` method):

```python
op_map = {
    # ...existing...
    "my_operation": self._my_operation,
}
```

3. **Update planner prompt** (`services/llm/prompts.py`) to include the new operation in the LLM's vocabulary.

### Adding a Query Handler

1. **Create handler** (`services/queries/handlers.py`):

```python
class MyQueryHandler(BaseQueryHandler):
    @property
    def query_type(self) -> str:
        return "my_query"
    
    @property
    def supported_functions(self) -> List[str]:
        return ["my_function"]
    
    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> QueryResult:
        # Implementation
        return self._success(result=...)
```

2. **Register** in `services/queries/registry.py` (add to `_register_default_handlers`).

3. **Add tool definition** in `services/llm/tools.py`.

---

## Design Decisions

**Why LangGraph over raw async chains?**
- Explicit state machine semantics
- Conditional edges without callback spaghetti
- Framework supports checkpointing for future pause/resume capability
- Visual graph debugging

**Why OpenRouter?**
- Model flexibility without vendor lock-in
- Cost optimization via model routing
- Free tier for development (Llama 3.3 70B)

**Why pgvector over dedicated vector DB?**
- Single database for all state
- Transactional consistency with metadata
- Sufficient scale for per-dataset embeddings

**Why immutable versions?**
- Full audit trail
- Safe rollback
- No destructive operations on source data

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Frontend | Next.js 14, React 18, TypeScript | App shell, data preview |
| API | FastAPI, Pydantic | REST endpoints, validation |
| Orchestration | LangGraph | Workflow state machine |
| LLM | OpenRouter (Llama 3.3 70B) | Intent, planning, chat |
| Data | Pandas | DataFrame operations |
| Database | PostgreSQL + pgvector | State, embeddings |
| Container | Docker Compose | Local/production deployment |

---

## License

MIT
