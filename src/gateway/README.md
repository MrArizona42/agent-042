# Gateway (FastAPI)

This service is the abstraction layer in front of the vLLM OpenAI-compatible server.

## Architecture

```mermaid
graph TD
    User([User/Client]) -->|POST /v1/chat/completions| FastAPI[FastAPI Gateway]
    User -->|GET /v1/models| FastAPI

    subgraph Gateway [Gateway Logic]
        FastAPI --> Routes[API Routes]
        Routes --> Processing[Processing Service]

        Processing --> Router[Task Router]
        Processing --> PromptBuilder[Prompt Builder]
        Processing --> VllmClient[vLLM Client]

        Router -.->|Decide Task| Processing
        PromptBuilder -.->|Build System Prompt| Processing
    end

    VllmClient -->|Proxy Request| vLLM[vLLM Server]
```

## Structure

- `main.py` — FastAPI app initialization
- `api/routes.py` — router aggregation
- `api/v1/discovery.py` — `/health` и `/config`
- `api/v1/openai_compat.py` — `/v1/models` и `/v1/chat/completions`
- `services/processing.py` — task routing, prompt building, RAG integration, sync/async execution

## Endpoints

- `GET /health`
- `GET /config`
- `GET /v1/models` (proxy)
- `POST /v1/chat/completions` (proxy + prompt/router layer)

### RAG Knowledge Base Selection

The `POST /v1/chat/completions` endpoint accepts an optional `knowledge_base` field
that controls which Qdrant collection is used for RAG retrieval:

| Value | Collection | Content |
|-------|-----------|---------|
| `null` | *(none)* | RAG disabled for this request |
| `"arxiv"` | `chat_documents` | ArXiv papers — ML / AI theory |
| `"pytorch_docs"` | `code_documents` | PyTorch documentation |

Example payload:

```json
{
  "messages": [{"role": "user", "content": "Explain attention mechanisms"}],
  "knowledge_base": "arxiv",
  "max_completion_tokens": 512
}
```

## Environment

- `GATEWAY_VLLM_BASE_URL` (default: `http://localhost:8000`)
- `GATEWAY_ASYNC_ENABLED` (default: `true`)
- `CELERY_BROKER_URL` and `REDIS_URL` (required when async mode is enabled)

## Run (local)

```bash
uv sync --extra gateway --extra worker --extra rag --extra dev
PYTHONPATH=src GATEWAY_ASYNC_ENABLED=false GATEWAY_VLLM_BASE_URL=http://localhost:8000 uvicorn gateway.main:app --reload --port 9000
```
