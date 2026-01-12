# UI (Streamlit)

Simple chat UI that talks to the FastAPI gateway.

## Environment

- `GATEWAY_URL` (default `http://localhost:9000`)

## Run (local)

```bash
uv sync
PYTHONPATH=src GATEWAY_URL=http://localhost:9000 streamlit run src/ui/app.py
```

