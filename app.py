import json
import os
import time
import uuid
from datetime import datetime

import requests
import streamlit as st


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_PROMPT = (
    "You are an expert content strategist assistant. "
    "Respond with practical, clear, high-impact guidance."
)
PROMPT_IMPROVEMENT_ACTIONS = [
    ("Clarify prompt", "Make my prompt clearer with concrete context and constraints."),
    ("Add structure", "Rewrite my prompt with clear sections and expected output format."),
    ("Make it specific", "Make my prompt more specific for stronger, actionable output."),
]
RESPONSE_CUSTOMIZATION_ACTIONS = [
    ("Shorter", "Make the response shorter and easier to skim."),
    ("More detailed", "Expand the response with deeper, practical detail."),
    ("More professional", "Rewrite the response in a more professional tone."),
    ("More creative", "Rewrite the response with more creative ideas."),
    ("Add examples", "Add relevant examples to strengthen the response."),
    ("Add CTA", "Add a strong call-to-action at the end."),
]


def initialize_state() -> None:
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv("OPENROUTER_API_KEY", "")
    if "model" not in st.session_state:
        st.session_state.model = "openai/gpt-4o-mini"
    if "chats" not in st.session_state:
        st.session_state.chats = []
    if "active_chat_id" not in st.session_state:
        new_chat()


def new_chat() -> None:
    chat_id = str(uuid.uuid4())
    st.session_state.chats.insert(
        0,
        {
            "id": chat_id,
            "title": "New chat",
            "created_at": datetime.now().isoformat(),
            "messages": [],
        },
    )
    st.session_state.active_chat_id = chat_id


def get_active_chat() -> dict:
    active_chat_id = st.session_state.active_chat_id
    for chat in st.session_state.chats:
        if chat["id"] == active_chat_id:
            return chat
    new_chat()
    return st.session_state.chats[0]


def update_chat_title_from_message(chat: dict, message: str) -> None:
    if chat["title"] != "New chat":
        return
    first_line = message.strip().splitlines()[0] if message.strip() else "New chat"
    chat["title"] = (first_line[:36] + "...") if len(first_line) > 39 else first_line


def generate_assistant_reply(*, api_key: str, model: str, messages: list[dict]) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Content.ai",
    }

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages[-20:]],
        "temperature": 0.7,
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    body_text = response.text or ""
    if response.status_code >= 400:
        raise RuntimeError(f"OpenRouter error ({response.status_code}): {body_text[:500]}")
    if not body_text.strip():
        raise RuntimeError("OpenRouter returned an empty response body.")

    try:
        result = response.json()
    except json.JSONDecodeError as exc:
        content_type = response.headers.get("Content-Type", "unknown")
        snippet = body_text[:500]
        raise RuntimeError(
            f"OpenRouter returned non-JSON response (content-type: {content_type}). Body: {snippet}"
        ) from exc

    choices = result.get("choices") or []
    if not choices:
        raise RuntimeError("No completion choices returned by OpenRouter.")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_value = part.get("text", "")
                if text_value:
                    text_parts.append(text_value)
        if text_parts:
            return "\n".join(text_parts).strip()

    raise RuntimeError("OpenRouter response did not include text content.")


def stream_text(text: str, placeholder) -> None:
    words = text.split()
    if len(words) < 30:
        placeholder.markdown(text)
        return

    chunk_size = max(1, len(words) // 80)
    rendered = []
    for index, word in enumerate(words, start=1):
        rendered.append(word)
        if index % chunk_size == 0 or index == len(words):
            placeholder.markdown(" ".join(rendered))
            time.sleep(0.01)


def request_assistant_update(*, chat: dict, messages: list[dict], action_label: str | None = None) -> None:
    with st.chat_message("assistant"):
        placeholder = st.empty()
        started_at = time.perf_counter()
        assistant_text = generate_assistant_reply(
            api_key=st.session_state.api_key.strip(),
            model=st.session_state.model.strip(),
            messages=messages,
        )
        thinking_time = time.perf_counter() - started_at
        stream_text(assistant_text, placeholder)
        action_suffix = f" · {action_label}" if action_label else ""
        st.markdown(
            f"<div class='chat-meta'>Thought for {thinking_time:.2f}s{action_suffix}</div>",
            unsafe_allow_html=True,
        )

    chat["messages"].append(
        {
            "role": "assistant",
            "content": assistant_text,
            "thinking_time": thinking_time,
            "action_label": action_label or "",
        }
    )


def last_message(chat: dict, role: str) -> str:
    for msg in reversed(chat["messages"]):
        if msg["role"] == role:
            return msg["content"]
    return ""


def render_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container {padding-top: 1.2rem; padding-bottom: 1.5rem;}
            [data-testid="stSidebar"] {border-right: 1px solid rgba(128, 128, 128, 0.2);}
            [data-testid="stChatMessage"] {
                border-radius: 14px;
                padding: 0.2rem 0.4rem;
            }
            .chat-meta {
                font-size: 0.8rem;
                color: #8b8b8b;
                margin-top: 0.3rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Content.ai", page_icon="AI", layout="wide")
    initialize_state()
    render_styles()

    with st.sidebar:
        st.title("Chats")
        st.button("New chat", on_click=new_chat, use_container_width=True, type="primary")
        st.divider()

        for chat in st.session_state.chats:
            is_active = chat["id"] == st.session_state.active_chat_id
            label = ("* " if is_active else "") + chat["title"]
            if st.button(label, key=f"chat-{chat['id']}", use_container_width=True):
                st.session_state.active_chat_id = chat["id"]
                st.rerun()

        st.divider()
        st.subheader("Settings")
        st.session_state.api_key = st.text_input(
            "OpenRouter API Key",
            value=st.session_state.api_key,
            type="password",
            help="Set OPENROUTER_API_KEY env var or paste the key here.",
        )
        st.session_state.model = st.text_input(
            "Model",
            value=st.session_state.model,
            help="Use any chat-capable model in your OpenRouter account.",
        )

    chat = get_active_chat()
    st.title(chat["title"] if chat["title"] != "New chat" else ":rainbow[Content.ai]")
    st.caption("A content strategist assistant.")

    if not chat["messages"]:
        st.info("Start by asking for ideas, rewrites, hooks, campaign concepts, or feedback.")

    for msg in chat["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "thinking_time" in msg:
                action_suffix = ""
                if msg.get("action_label"):
                    action_suffix = f" · {msg['action_label']}"
                st.markdown(
                    f"<div class='chat-meta'>Thought for {msg['thinking_time']:.2f}s{action_suffix}</div>",
                    unsafe_allow_html=True,
                )

    clicked_action = None
    if chat["messages"] and last_message(chat, "assistant"):
        st.divider()
        st.caption("Improve or customize this response")
        st.markdown("**Improve prompt**")
        prompt_columns = st.columns(len(PROMPT_IMPROVEMENT_ACTIONS))
        for index, action in enumerate(PROMPT_IMPROVEMENT_ACTIONS):
            with prompt_columns[index]:
                if st.button(
                    action[0],
                    key=f"prompt-action-{chat['id']}-{index}-{len(chat['messages'])}",
                    use_container_width=True,
                ):
                    clicked_action = ("prompt", action[0], action[1])

        st.markdown("**Customize response**")
        response_columns = st.columns(3)
        for index, action in enumerate(RESPONSE_CUSTOMIZATION_ACTIONS):
            with response_columns[index % 3]:
                if st.button(
                    action[0],
                    key=f"response-action-{chat['id']}-{index}-{len(chat['messages'])}",
                    use_container_width=True,
                ):
                    clicked_action = ("response", action[0], action[1])

    if clicked_action:
        if not st.session_state.api_key.strip():
            st.error("Please provide your OpenRouter API key in the sidebar settings.")
            return

        action_type, action_label, action_instruction = clicked_action
        last_user_prompt = last_message(chat, "user")
        if action_type == "prompt":
            instruction = (
                f"Your task: {action_instruction}\n"
                f"Original user prompt:\n{last_user_prompt}\n\n"
                "First output one line starting with 'Improved prompt:'. "
                "Then output 'Improved response:' and provide the improved answer."
            )
        else:
            instruction = (
                "Revise your latest response using this direction:\n"
                f"{action_instruction}\n"
                "Return only the improved response."
            )

        try:
            request_assistant_update(
                chat=chat,
                messages=[*chat["messages"], {"role": "user", "content": instruction}],
                action_label=action_label,
            )
        except (requests.RequestException, RuntimeError) as exc:
            st.error(str(exc))
            return
        st.rerun()

    user_prompt = st.chat_input("Message your assistant...")
    if not user_prompt:
        return

    if not st.session_state.api_key.strip():
        st.error("Please provide your OpenRouter API key in the sidebar settings.")
        return

    user_message = {"role": "user", "content": user_prompt.strip()}
    chat["messages"].append(user_message)
    update_chat_title_from_message(chat, user_prompt)

    with st.chat_message("user"):
        st.markdown(user_prompt)

    try:
        request_assistant_update(
            chat=chat,
            messages=chat["messages"],
        )
    except (requests.RequestException, RuntimeError) as exc:
        st.error(str(exc))
        return
    st.rerun()


if __name__ == "__main__":
    main()
