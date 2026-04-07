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
- `GET /v1/knowledge-bases` — list available KBs, aliases, and per-alias query config
- `POST /v1/admin/reload-config` — hot-reload `knowledge_bases.json` (requires authenticated session)

### RAG Knowledge Base Selection

The `POST /v1/chat/completions` endpoint accepts an optional `rag_sources` array
that controls which Qdrant collections are used for RAG retrieval:

| Field | Description |
|-------|-------------|
| `knowledge_base` | KB name, e.g. `"arxiv"`, `"pytorch_docs"` |
| `alias` | Alias role, e.g. `"champion"`, `"challenger"`. Uses the KB's `default_alias` when `null`. |

Example payload:

```json
{
  "messages": [{"role": "user", "content": "Explain attention mechanisms"}],
  "rag_sources": [{"knowledge_base": "arxiv", "alias": "challenger"}],
  "max_completion_tokens": 512
}
```

### Alias-Owned Query Config

Query-time RAG parameters (`top_k`, `score_threshold`, `context_max_length`,
`reranker`) are set per alias in `knowledge_bases.json`, not via environment
variables. To change retrieval behavior:

1. Edit the alias entry in `knowledge_bases.json`
2. `POST /v1/admin/reload-config` (requires authenticated session)
3. Next request with that alias uses the new config immediately

## Environment

Shared endpoint vars use canonical names. Gateway-specific behavior keeps the
`GATEWAY_` prefix.

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_BASE_URL` | `http://localhost:8000` | Shared vLLM server URL |
| `QDRANT_HOST` | `localhost` | Shared Qdrant host |
| `QDRANT_PORT` | `6333` | Shared Qdrant port |
| `EMBEDDINGS_URL` | `http://localhost:8100` | Shared embeddings service URL |
| `GATEWAY_RAG_ENABLED` | `true` | Enable/disable RAG |
| `GATEWAY_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `GATEWAY_ASYNC_ENABLED` | `true` | Enable Celery async mode |
| `GATEWAY_RAG_STRICT_STARTUP` | `false` | Raise on legacy/invalid Qdrant collections at startup |

Query-time RAG parameters (`top_k`, `score_threshold`, `context_max_length`)
are no longer environment variables. They are now alias-owned config in
`knowledge_bases.json`. The old `GATEWAY_TOP_K` and `GATEWAY_SCORE_THRESHOLD`
env vars are retired.

See `src/shared/config.py` for the full list.

## Run (local)

`gateway.main` explicitly loads the repo-root `.env` for local runs before
settings are cached.

```bash
uv sync --extra gateway --extra worker --extra rag --group dev
PYTHONPATH=src uvicorn gateway.main:app --reload --port 9000
```
