# UI (Streamlit)

Simple chat UI that talks to the FastAPI gateway.

## Features

- Chat interface with two-channel streaming: thinking expander + live answer body
- Full prompt expander populated from gateway prompt preview
- **Knowledge Base selector** — choose which RAG collection to query:
  - *Disabled* — no retrieval
  - *ArXiv papers (ML / AI theory)* — latest ML/AI research
  - *PyTorch docs (coding)* — PyTorch API documentation

## Gateway Contract

The UI uses the gateway's rich first-party stream:

- sends `X-UI-Rich-Stream: 1` on `POST /v1/chat/completions`
- reads `X-Request-Id` from the streaming response
- fetches `GET /v1/chat/prompt-preview/{request_id}` to show the full prompt and RAG context
- renders `thinking_token` and `answer_token` events in separate UI containers

## Environment

- `GATEWAY_URL` (default `http://localhost:9001` for host-side runs; Compose injects `http://gateway:9000` inside the UI container)

## Run (local)

`src/ui/app.py` explicitly loads the repo-root `.env` for local runs before
settings are cached.

```bash
uv sync --extra ui --group dev
PYTHONPATH=src streamlit run src/ui/app.py
```

## Production Deployment

The UI is deployed behind nginx at `https://agent.antonlab.ru:8443`.

### Endpoints

| Endpoint | Service | Description |
|----------|---------|-------------|
| `/` | Streamlit UI | Main chat interface |
| `/_stcore/stream` | Streamlit UI | WebSocket for real-time updates |
| `/api/*` | Gateway | REST API (stripped to `/` when forwarded) |
| `/api/v1/chat/completions` | Gateway | SSE chat endpoint used with `X-UI-Rich-Stream: 1` |
| `/api/v1/chat/prompt-preview/{request_id}` | Gateway | Prompt preview for the active streamed request |
| `/docs` | Gateway | Swagger/OpenAPI documentation |
| `/openapi.json` | Gateway | OpenAPI schema |
| `/health` | Gateway | Health check endpoint |

### Configuration Files

- **Streamlit config**: `.streamlit/config.toml` - CORS, domain, and server settings
- **Nginx config**: `infra/nginx/agent.antonlab.ru.conf` - Reverse proxy configuration

### Setup

```bash
# 1. Install nginx config
sudo cp infra/nginx/agent.antonlab.ru.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/agent.antonlab.ru.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 2. Get SSL certificate
sudo certbot --nginx -d agent.antonlab.ru

# 3. Rebuild and start UI container
cd infra/compose
docker compose build ui
docker compose up -d ui
```
