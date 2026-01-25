# UI (Streamlit)

Simple chat UI that talks to the FastAPI gateway.

## Environment

- `GATEWAY_URL` (default `http://localhost:9000`)

## Run (local)

```bash
uv sync
PYTHONPATH=src GATEWAY_URL=http://localhost:9000 streamlit run src/ui/app.py
```

## Production Deployment

The UI is deployed behind nginx at `https://agent.antonlab.ru:8443`.

### Endpoints

| Endpoint | Service | Description |
|----------|---------|-------------|
| `/` | Streamlit UI | Main chat interface |
| `/_stcore/stream` | Streamlit UI | WebSocket for real-time updates |
| `/api/*` | Gateway | REST API (stripped to `/` when forwarded) |
| `/api/v1/chat/completions` | Gateway | OpenAI-compatible chat endpoint |
| `/docs` | Gateway | Swagger/OpenAPI documentation |
| `/openapi.json` | Gateway | OpenAPI schema |
| `/health` | Gateway | Health check endpoint |

### Configuration Files

- **Streamlit config**: `.streamlit/config.toml` - CORS, domain, and server settings
- **Nginx config**: `infra/nginx/agent.antonlab.ru.conf` - Reverse proxy configuration

### Setup

```bash

# 2. Install nginx config
sudo cp infra/nginx/agent.antonlab.ru.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/agent.antonlab.ru.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 1. Get SSL certificate
sudo certbot --nginx -d agent.antonlab.ru

# 3. Rebuild and start UI container
docker compose build ui
docker compose up -d ui
```
