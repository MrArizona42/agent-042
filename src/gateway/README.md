# Gateway (FastAPI)

This service is the abstraction layer in front of the vLLM OpenAI-compatible server.
It provides task routing, prompt building, RAG integration, and optional async
execution via Celery + Redis.

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
        Processing --> RAGService[RAG Service]
        Processing --> VllmClient[vLLM Client]
        Processing --> CeleryClient[Celery Client]

        Router -.->|Decide Task| Processing
        PromptBuilder -.->|Build System Prompt| Processing
        RAGService -.->|Retrieve Context| Processing
    end

    VllmClient -->|Proxy Request| vLLM[vLLM Server]
    RAGService -->|Query| Qdrant[Qdrant Vector DB]
    CeleryClient -->|Dispatch Task| RabbitMQ[RabbitMQ]
    CeleryClient -.->|Stream Tokens| Redis[Redis Pub/Sub]
```

## Structure & Classes

```mermaid
classDiagram
    class FastAPIApp {
        <<Entrypoint>>
        main.py
    }
    class APIRoutes {
        <<Router>>
        routes.py
        openai_compat.py
        discovery.py
    }
    class ProcessingService {
        <<Service>>
        processing.py
        +process_chat
    }
    class TaskRouter {
        <<Logic>>
        task_router.py
        +decide(text)
    }
    class PromptBuilder {
        <<Logic>>
        prompt_builder.py
        +build_system_prompt(task)
    }
    class RAGService {
        <<Service>>
        rag_service.py
        +retrieve(query, kb)
    }
    class vLLMClient {
        <<Client>>
        vllm_client.py
        +chat_completions(payload)
    }
    class CeleryClient {
        <<Client>>
        celery_client.py
        +dispatch(task)
    }
    class RedisStreamService {
        <<Client>>
        redis_stream.py
        +subscribe(channel)
    }

    FastAPIApp --> APIRoutes : includes
    APIRoutes --> ProcessingService : calls
    ProcessingService --> TaskRouter : uses
    ProcessingService --> PromptBuilder : uses
    ProcessingService --> RAGService : uses
    ProcessingService --> vLLMClient : uses
    ProcessingService --> CeleryClient : uses
    CeleryClient --> RedisStreamService : reads tokens
```

## Endpoints

- `GET /health`
- `GET /config`
- `GET /v1/models` (proxy)
- `POST /v1/chat/completions` (proxy + prompt/router layer; supports `stream: true`)

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

Key environment variables (all prefixed `GATEWAY_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_VLLM_BASE_URL` | `http://localhost:8000` | vLLM server URL |
| `GATEWAY_QDRANT_HOST` | `localhost` | Qdrant host |
| `GATEWAY_QDRANT_PORT` | `6333` | Qdrant port |
| `GATEWAY_RAG_ENABLED` | `true` | Enable/disable RAG |
| `GATEWAY_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `GATEWAY_ASYNC_ENABLED` | `true` | Enable Celery async mode |
| `GATEWAY_TOP_K` | `5` | Number of RAG documents to retrieve |
| `GATEWAY_SCORE_THRESHOLD` | `0.0` | Minimum similarity score for RAG |

See `src/shared/config.py` for the full list.

## Run (local)

```bash
uv sync --extra gateway --extra worker --extra rag --extra dev
PYTHONPATH=src GATEWAY_VLLM_BASE_URL=http://localhost:8000 uvicorn gateway.main:app --reload --port 9000
```
