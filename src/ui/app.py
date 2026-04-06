from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from shared.config import bootstrap_local_settings_env, get_knowledge_bases
from ui.client import GatewayClient
from ui.config import get_settings

bootstrap_local_settings_env(repo_root=Path(__file__).resolve().parents[2])

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
            with st.expander("💭 Thinking...", expanded=False):
                st.markdown(part_content)
        else:
            text = part_content.strip()
            if text:
                st.markdown(text)


st.title("agent-042")
st.caption("Streamlit UI → FastAPI Gateway → vLLM (OpenAI-compatible)")

# Get settings (cached)
settings = get_settings()

gateway_url = settings.url

# Forward the browser's session cookie to the Gateway backend
_browser_session_id = st.context.cookies.get("session_id")
client = GatewayClient(gateway_url, session_id=_browser_session_id)

# ------------------------------------------------------------------
# Auth check — redirect to /auth/login if not authenticated
# ------------------------------------------------------------------
user_info = client.me()

if user_info is None:
    st.info("You need to sign in to use agent-042.")
    st.link_button("Log in with Google", "/auth/login")
    st.stop()

with st.sidebar:
    if user_info:
        # Show user info + logout
        col1, col2 = st.columns([1, 3])
        if user_info.get("picture"):
            col1.image(user_info["picture"], width=40)
        col2.markdown(f"**{user_info.get('name', 'User')}**")
        st.link_button("Logout", "/auth/logout")

        st.divider()

        # ---- Chat sessions ----
        st.subheader("Chat Sessions")
        if st.button("➕ New Chat"):
            st.session_state.pop("chat_session_id", None)
            st.session_state.messages = []
            st.rerun()

        try:
            sessions = client.list_chat_sessions()
            for sess in sessions:
                label = sess.get("title") or "Untitled"
                col_name, col_del = st.columns([5, 1])
                with col_name:
                    if st.button(label, key=f"sess_{sess['id']}", use_container_width=True):
                        st.session_state.chat_session_id = sess["id"]
                        msgs = client.get_session_messages(sess["id"])
                        st.session_state.messages = [
                            {"role": m["role"], "content": m["content"]} for m in msgs
                        ]
                        st.rerun()
                with col_del:
                    if st.button("🗑️", key=f"del_{sess['id']}"):
                        st.session_state.confirm_delete_id = sess["id"]

                # Confirmation row appears below the session entry
                if st.session_state.get("confirm_delete_id") == sess["id"]:
                    st.warning(f"Delete **{label}**?")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Yes, delete", key=f"yes_{sess['id']}"):
                            try:
                                client.delete_chat_session(sess["id"])
                                # If we deleted the active session, reset
                                if st.session_state.get("chat_session_id") == sess["id"]:
                                    st.session_state.pop("chat_session_id", None)
                                    st.session_state.messages = []
                            except Exception as e:
                                st.error(f"Failed to delete session: {e}")
                            st.session_state.pop("confirm_delete_id", None)
                            st.rerun()
                    with c2:
                        if st.button("Cancel", key=f"no_{sess['id']}"):
                            st.session_state.pop("confirm_delete_id", None)
                            st.rerun()
        except Exception as e:
            st.warning(f"Could not load chat sessions: {e}")

        st.divider()

    st.subheader("Knowledge Base")

    # Build options from the knowledge base registry
    kb_options: dict[str, str | None] = {"Disabled": None}
    _kb_meta: dict[str, dict] = {}
    for task_cfg in get_knowledge_bases().values():
        for kb_cfg in task_cfg.knowledge_bases:
            kb_options[kb_cfg.label] = kb_cfg.name
            _kb_meta[kb_cfg.name] = {"description": kb_cfg.description}

    selected_kb_label = st.radio(
        "Select knowledge base for RAG retrieval",
        options=list(kb_options.keys()),
        index=0,
    )
    selected_kb = kb_options[selected_kb_label]

    if selected_kb:
        st.caption(_kb_meta[selected_kb]["description"])


# ------------------------------------------------------------------
# Chat session — created lazily on first message (avoids empty sessions)
# ------------------------------------------------------------------

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

    # Lazily create a chat session on first message
    if not st.session_state.get("chat_session_id"):
        try:
            sess = client.create_chat_session()
            st.session_state.chat_session_id = sess["id"]
        except Exception as e:
            st.warning(f"Could not create chat session: {e}")
            st.session_state.chat_session_id = None

    payload = {
        # "model": None,
        "messages": st.session_state.messages,
        "max_completion_tokens": settings.max_completion_tokens,
        "stream": False,
    }
    if selected_kb:
        payload["rag_sources"] = [{"knowledge_base": selected_kb}]
    if st.session_state.get("chat_session_id"):
        payload["chat_session_id"] = st.session_state.chat_session_id

    with st.chat_message("assistant"):
        try:
            resp = client.chat(payload)
            content = resp["choices"][0]["message"]["content"]

            # Show the full prompt sent to the LLM (system prompt + RAG context)
            prompt_messages = resp.get("_prompt_messages")
            if prompt_messages:
                parts = []
                for pm in prompt_messages:
                    role = pm.get("role", "unknown").upper()
                    body = pm.get("content", "")
                    parts.append(f"**[{role}]**\n\n{body}\n\n---\n\n")
                with st.expander("📋 Full prompt", expanded=False):
                    st.markdown("".join(parts))

        except Exception as e:
            content = f"Error: {e}"
        render_message_with_thinking(content)

    st.session_state.messages.append({"role": "assistant", "content": content})
