# Dataset Curator

An AI-powered tool for cleaning and transforming CSV datasets through natural language. Built with FastAPI, Next.js, and LangGraph.

```
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|  Upload CSV      | --> |  Chat with AI    | --> |  Download Clean  |
|                  |     |                  |     |  Dataset         |
+------------------+     +------------------+     +------------------+
```

## Features

- **Natural Language Transformations** - "remove nulls and sort by date"
- **Multi-Step Execution** - Complex operations broken into atomic steps
- **Automatic Retries** - Failed steps retry automatically
- **Version Control** - Every transformation creates a new version
- **Data Querying** - Ask questions about your data

## Architecture

```
+-------------------+          +-------------------+          +------------+
|                   |   HTTP   |                   |   SQL    |            |
|  Next.js Frontend |<-------->|  FastAPI Backend  |<-------->| PostgreSQL |
|  (React)          |          |  (LangGraph)      |          |            |
+-------------------+          +-------------------+          +------------+
        |                              |
        |                              |
        v                              v
+-------------------+          +-------------------+
|                   |          |                   |
|  Data Preview     |          |  OpenRouter LLM   |
|  (Paginated)      |          |  (Llama 3.3 70B)  |
+-------------------+          +-------------------+
```

## Request Flow

```
User Message: "remove nulls, drop age column, sort by name"
                              |
                              v
                    +-------------------+
                    | Intent Classifier |
                    | (transform/chat)  |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
    +------------------+          +------------------+
    | LangGraph Planner|          |   Chat Agent     |
    | (creates steps)  |          | (answers queries)|
    +--------+---------+          +------------------+
             |
             v
    +------------------+
    |  Execution Graph |
    +------------------+
             |
    +--------+--------+--------+
    |        |        |        |
    v        v        v        v
 +------+ +------+ +------+ +------+
 | Plan | | Load | |Execute| |Final-|
 |      | | Data | | Steps | | ize  |
 +------+ +------+ +------+ +------+
                      |
              +-------+-------+
              |       |       |
              v       v       v
           Step 1  Step 2  Step 3
           (retry) (retry) (retry)
              |       |       |
              v       v       v
           Validate Validate Validate
```

## LangGraph Workflow

The transformation workflow uses LangGraph for orchestration:

```
                         +-------+
                         | START |
                         +---+---+
                             |
                             v
                        +----+----+
                        |  PLAN   |
                        | (async) |
                        +----+----+
                             |
                             v
                      +------+------+
                      | LOAD DATA   |
                      +------+------+
                             |
                             v
                    +--------+--------+
                    | CHECK APPROVAL  |
                    +--------+--------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
      [needs approval]              [auto-approve]
              |                             |
              v                             v
        +-----+-----+               +-------+-------+
        | FINALIZE  |               | EXECUTE STEP  |
        +-----------+               +-------+-------+
                                            |
                                            v
                                    +-------+-------+
                                    |   VALIDATE    |
                                    +-------+-------+
                                            |
                          +-----------------+------------------+
                          |                 |                  |
                          v                 v                  v
                      [success]         [failed]          [max retries]
                          |                 |                  |
                          v                 v                  v
                   +------+------+   +------+------+    +------+------+
                   | Next Step?  |   |    RETRY    |    | Skip Step   |
                   +------+------+   +-------------+    +------+------+
                          |                                    |
              +-----------+-----------+                        |
              |                       |                        |
              v                       v                        |
         [more steps]            [no more]                     |
              |                       |                        |
              v                       v                        |
      CHECK APPROVAL             FINALIZE <--------------------+
                                     |
                                     v
                                  +--+--+
                                  | END |
                                  +-----+
```

## Project Structure

```
dataset-curator/
|
+-- backend/
|   +-- agents/
|   |   +-- data_ops.py        # DataFrame operations (15+ transforms)
|   |
|   +-- api/
|   |   +-- routes.py          # REST endpoints
|   |
|   +-- orchestrator/
|   |   +-- workflow.py        # LangGraph workflow (unified)
|   |
|   +-- models/
|   |   +-- dataset_state.py   # Pydantic state model
|   |   +-- db_models.py       # SQLAlchemy models
|   |
|   +-- llm.py                 # Intent classification + planning
|   +-- db.py                  # Database connection
|   +-- main.py                # FastAPI app
|
+-- frontend/
|   +-- app/
|   |   +-- page.tsx           # Main chat + preview UI
|   |   +-- globals.css        # Styles
|   |
|   +-- components/
|   |   +-- sidebar.tsx        # Session history
|   |   +-- ui/                # Reusable components
|   |
|   +-- lib/
|   |   +-- api.ts             # API client
|
+-- storage/
|   +-- raw/                   # Uploaded files
|   +-- curated/               # Transformed versions
|
+-- docker-compose.yml
```

## Available Operations

| Operation | Example | Description |
|-----------|---------|-------------|
| `drop_column` | "remove the age column" | Delete a column |
| `rename_column` | "rename id to user_id" | Rename a column |
| `drop_nulls` | "remove rows with missing values" | Drop null rows |
| `fill_nulls` | "fill nulls in age with 0" | Replace nulls |
| `drop_duplicates` | "remove duplicate rows" | Deduplicate |
| `filter_rows` | "keep only rows where age > 18" | Filter (keep) |
| `drop_rows` | "remove rows where status = inactive" | Filter (remove) |
| `add_column` | "add column country with value USA" | Add static column |
| `add_conditional_column` | "add difficulty: <30 Hard, 30-60 Medium, >60 Easy" | Add computed column |
| `convert_type` | "convert age to integer" | Change dtype |
| `sort` | "sort by date descending" | Sort rows |
| `replace_values` | "replace NA with Unknown in status" | Find/replace |
| `lowercase` | "lowercase the name column" | To lowercase |
| `uppercase` | "uppercase the code column" | To uppercase |
| `strip_whitespace` | "trim whitespace from all text" | Strip spaces |

## API Endpoints

```
POST /upload
     |-- Upload CSV file
     +-- Returns: preview, row_count, column_count

POST /chat/{dataset_id}
     |-- Send natural language message
     +-- Returns: assistant response

GET  /preview/{dataset_id}?page=1&page_size=50
     |-- Get paginated data preview
     +-- Returns: data, pagination info

GET  /download/{dataset_id}/file
     |-- Download processed CSV
     +-- Returns: file download
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenRouter API key (get one at https://openrouter.ai)

### Setup

1. Clone and configure:
```bash
git clone <repo>
cd dataset-curator

# Create .env file with required settings
cat > .env << 'EOF'
OPENROUTER_API_KEY=your_key_here
DATABASE_URL=postgresql+asyncpg://dataset_curator:your_secure_password@db:5432/dataset_curator
EOF
```

2. Start services:
```bash
docker compose up --build
```

3. Access the app:
```
Frontend:  http://localhost:3000
API:       http://localhost:8000
API Docs:  http://localhost:8000/docs
```

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string (e.g., `postgresql+asyncpg://user:pass@host:5432/db`) |
| `OPENROUTER_API_KEY` | API key from [OpenRouter](https://openrouter.ai/keys) |

### Optional - LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_LLM_MODEL` | `meta-llama/llama-3.3-70b-instruct:free` | LLM model for transformations |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API base URL |

### Optional - Embedding Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model for semantic search |
| `EMBEDDING_DIM` | `384` | Embedding vector dimension (must match model) |

### Optional - Storage Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RAW_STORAGE_PATH` | `storage/raw` | Path for uploaded files |
| `CURATED_STORAGE_PATH` | `storage/curated` | Path for processed files |

### Optional - Data Processing Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `LARGE_FILE_SIZE_MB` | `100` | Files larger are considered "large" |
| `LARGE_ROW_COUNT` | `1000000` | Datasets with more rows are "large" |
| `DATA_SAMPLE_SIZE` | `10000` | Sample size for large file queries |
| `DATA_CHUNK_SIZE` | `50000` | Chunk size for streaming operations |

### Optional - CORS Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated allowed origins |

### Optional - Approval Flow

| Variable | Default | Description |
|----------|---------|-------------|
| `REQUIRE_APPROVAL_FOR_DESTRUCTIVE_OPS` | `false` | Require approval for destructive ops |
| `APPROVAL_ROW_THRESHOLD` | `1000` | Row threshold triggering approval |

## Development

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Tests
```bash
cd backend
pytest
```

## Tech Stack

```
+------------------+------------------+------------------+
|     Frontend     |     Backend      |   Infrastructure |
+------------------+------------------+------------------+
| Next.js 14       | FastAPI          | Docker           |
| React 18         | LangGraph        | PostgreSQL       |
| TypeScript       | Pandas           | OpenRouter       |
|                  | SQLAlchemy       | Llama 3.3 70B    |
+------------------+------------------+------------------+
```

## Example Usage

```
User: "Upload sales.csv"
Bot:  Dataset loaded! 1000 rows, 8 columns.

User: "Remove rows where revenue is null and sort by date descending"
Bot:  Executing 2 steps:
      
      Step 1: Remove rows with null revenue
        [OK] Removed 45 rows. Now 955 rows.
      
      Step 2: Sort by date descending  
        [OK] Sorted by date (descending)
      
      Summary: 2/2 steps completed
      Result: 955 rows x 8 columns

User: "What's the average revenue?"
Bot:  The average revenue is $1,234.56 across 955 records.

User: "Download"
Bot:  [Downloads sales_v2.csv]
```

## License

MIT
