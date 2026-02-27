from __future__ import annotations

import re

import streamlit as st

from shared.config import KNOWLEDGE_BASES
from ui.client import GatewayClient
from ui.config import get_settings

st.set_page_config(page_title="agent-042", layout="wide")


def render_message_with_thinking(content: str) -> None:
    """Render message content, styling <think> blocks separately."""
    # Pattern to match <think>...</think> blocks
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    parts = []
    last_end = 0

    for match in think_pattern.finditer(content):
        # Add text before the think block
        if match.start() > last_end:
            parts.append(("text", content[last_end : match.start()]))
        # Add the think block content
        parts.append(("think", match.group(1).strip()))
        last_end = match.end()

    # Add remaining text after last think block
    if last_end < len(content):
        parts.append(("text", content[last_end:]))

    # Render each part
    for part_type, part_content in parts:
        if part_type == "think":
            st.markdown(
                f"""<div style="
                    background-color: rgba(128, 128, 128, 0.1);
                    border-left: 3px solid rgba(128, 128, 128, 0.4);
                    padding: 8px 12px;
                    margin: 8px 0;
                    border-radius: 4px;
                    font-size: 0.85em;
                    color: rgba(150, 150, 150, 1);
                "><em>💭 Thinking...</em><br/>{part_content}</div>""",
                unsafe_allow_html=True,
            )
        else:
            text = part_content.strip()
            if text:
                st.markdown(text)


st.title("agent-042")
st.caption("Streamlit UI → FastAPI Gateway → vLLM (OpenAI-compatible)")

# Get settings (cached)
settings = get_settings()

gateway_url = settings.url
client = GatewayClient(gateway_url)

with st.sidebar:
    st.markdown("---")
    st.subheader("Model Settings")
    max_tokens = st.number_input("max_tokens", min_value=1, value=512, step=1)

    st.markdown("---")
    st.subheader("Knowledge Base")

    # Build options from the KNOWLEDGE_BASES registry
    kb_options: dict[str, str | None] = {"Disabled": None}
    for kb_key, kb_info in KNOWLEDGE_BASES.items():
        kb_options[kb_info["label"]] = kb_key

    selected_kb_label = st.radio(
        "Select knowledge base for RAG retrieval",
        options=list(kb_options.keys()),
        index=0,
    )
    selected_kb = kb_options[selected_kb_label]

    if selected_kb:
        st.caption(KNOWLEDGE_BASES[selected_kb]["description"])


if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            render_message_with_thinking(m["content"])
        else:
            st.markdown(m["content"])

prompt = st.chat_input("Ask something")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        # "model": None,
        "messages": st.session_state.messages,
        "max_completion_tokens": int(max_tokens),
        "stream": False,
        "knowledge_base": selected_kb,
    }

    with st.chat_message("assistant"):
        try:
            resp = client.chat(payload)
            content = resp["choices"][0]["message"]["content"]
        except Exception as e:
            content = f"Error: {e}"
        render_message_with_thinking(content)

    st.session_state.messages.append({"role": "assistant", "content": content})

# with st.expander("Raw messages"):
#     st.code(json.dumps(st.session_state.messages, ensure_ascii=False, indent=2))
