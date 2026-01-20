from __future__ import annotations

import json

import streamlit as st

from ui.client import GatewayClient
from ui.config import get_settings

st.set_page_config(page_title="agent-042", layout="wide")

st.title("agent-042")
st.caption("Streamlit UI → FastAPI Gateway → vLLM (OpenAI-compatible)")

# Get settings (cached)
settings = get_settings()

with st.sidebar:
    gateway_url = st.text_input("Gateway URL", value=settings.url)
    client = GatewayClient(gateway_url)

    if st.button("Check health"):
        try:
            st.success(client.health())
        except Exception as e:
            st.error(str(e))

    st.markdown("---")
    st.subheader("Model Settings")
    model = st.text_input("Model (optional)", value="")
    temperature = st.slider("temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.05)
    max_tokens = st.number_input("max_tokens", min_value=1, value=512, step=1)

    st.markdown("---")
    st.subheader("RAG Settings")
    st.info(
        "RAG is automatically enabled on the gateway."
        " The system will retrieve relevant context from the knowledge base based on your query."
    )
    st.caption("Available collections: chat (ArXiv papers), code (PyTorch docs)")


if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask something")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "model": model or None,
        "messages": st.session_state.messages,
        "temperature": temperature,
        "max_completion_tokens": int(max_tokens),
        "stream": False,
    }

    with st.chat_message("assistant"):
        try:
            resp = client.chat(payload)
            content = resp["choices"][0]["message"]["content"]
        except Exception as e:
            content = f"Error: {e}"
        st.markdown(content)

    st.session_state.messages.append({"role": "assistant", "content": content})

with st.expander("Raw messages"):
    st.code(json.dumps(st.session_state.messages, ensure_ascii=False, indent=2))
