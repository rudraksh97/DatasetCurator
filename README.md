# Agentic Dataset Curator

An AI-powered tool for ingesting, analyzing, cleaning, and documenting datasets. Built with FastAPI (backend), Next.js (frontend), and LLM integration via OpenRouter.

## Features

- **Dataset Ingestion** - Upload CSV files for analysis
- **Schema Inference** - Automatically detect column types and structure
- **Quality Analysis** - Identify missing values, duplicates, outliers, type mismatches
- **AI Chat Interface** - Interact with your data through natural language
- **Automated Cleaning** - Apply fixes with human-in-the-loop approval
- **Dataset Versioning** - Track all transformations with immutable versions
- **Documentation Generation** - Auto-generate dataset cards

## Project Structure

```
├── backend/               # Python API service
│   ├── agents/           # Data processing agents
│   ├── api/              # FastAPI endpoints
│   ├── models/           # Pydantic & SQLAlchemy models
│   ├── orchestrator/     # Agent coordination (LangGraph)
│   ├── tests/            # Backend tests
│   ├── db.py             # Database config
│   ├── llm.py            # OpenRouter LLM client
│   ├── main.py           # App entry
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/             # Next.js app
│   ├── app/              # Pages & layouts
│   ├── components/       # UI components
│   │   ├── sidebar.tsx   # Chat history sidebar
│   │   └── ui/           # shadcn-style components
│   ├── lib/              # API client
│   ├── types/            # TypeScript types
│   └── Dockerfile
├── storage/
│   ├── raw/              # Uploaded datasets
│   └── curated/          # Processed datasets
├── examples/             # Sample data files
├── docker-compose.yml
└── README.md
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- OpenRouter API key

### Setup

1. **Create environment file:**
   ```bash
   cat > .env << 'EOF'
   OPENROUTER_API_KEY=your_openrouter_key_here
   DATABASE_URL=postgresql+asyncpg://dataset_curator:dataset_curator@localhost:5432/dataset_curator
   EOF
   ```

2. **Start services:**
   ```bash
   docker compose up --build
   ```

3. **Access the app:**
   - Frontend: http://localhost:3000
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload and process a dataset |
| GET | `/health/{dataset_id}` | Get quality issues report |
| POST | `/approve` | Approve and apply fixes |
| GET | `/download/{dataset_id}/file` | Download processed dataset |
| POST | `/chat/{dataset_id}` | Send message to AI agent |
| GET | `/chat/{dataset_id}` | Get chat history |
| POST | `/analyze/{dataset_id}` | Get AI analysis of dataset |
| GET | `/card/{dataset_id}` | Get dataset documentation |

## Development

### Local Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Local Frontend
```bash
cd frontend
npm install
npm run dev
```

### Run Tests
```bash
cd backend
pytest
```

## Tech Stack

- **Backend:** Python, FastAPI, SQLAlchemy, Pandas
- **Frontend:** Next.js 14, React, TypeScript
- **Database:** PostgreSQL
- **LLM:** OpenRouter (Claude 3.5 Sonnet)
- **Containerization:** Docker

## License

MIT
