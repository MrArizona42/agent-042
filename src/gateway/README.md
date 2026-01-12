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
    class vLLMClient {
        <<Client>>
        vllm_client.py
        +chat_completions(payload)
    }

    FastAPIApp --> APIRoutes : includes
    APIRoutes --> ProcessingService : calls
    ProcessingService --> TaskRouter : uses
    ProcessingService --> PromptBuilder : uses
    ProcessingService --> vLLMClient : uses
```

## Endpoints

- `GET /health`
- `GET /config`
- `GET /v1/models` (proxy)
- `POST /v1/chat/completions` (proxy + prompt/router layer)

## Environment

- `GATEWAY_VLLM_BASE_URL` (default: `http://localhost:8000`)

## Run (local)

```bash
uv sync
PYTHONPATH=src GATEWAY_VLLM_BASE_URL=http://localhost:8000 uvicorn gateway.main:app --reload --port 9000
```

