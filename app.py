import json
import os
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

import streamlit as st  # type: ignore[reportMissingImports]


def load_app_title() -> str:
    return os.getenv("INUKTITUT_APP_TITLE", "Chat Interface")


def load_assistant_greeting() -> str:
    return os.getenv("INUKTITUT_ASSISTANT_GREETING", "Start a conversation.")


st.set_page_config(page_title=load_app_title(), page_icon="ᐃ", layout="wide")


@dataclass(frozen=True)
class AppConfig:
    backend_url: str
    timeout_seconds: int
    models_url: str
    chat_store_path: Path


APP_TITLE = load_app_title()
APP_CAPTION = os.getenv("INUKTITUT_APP_CAPTION", "Backend-connected multi-chat UI")
ASSISTANT_GREETING = load_assistant_greeting()
DEFAULT_MODELS = [
    {"id": "base", "label": "Base model", "endpoint": "/generate", "model_type": "base"},
    {"id": "adapted", "label": "Adapted model", "endpoint": "/generate", "model_type": "adapted"},
]
APP_ROOT = Path(__file__).resolve().parent
DEFAULT_CHAT_STORE_PATH = APP_ROOT / "chat_history.json"
DEFAULT_CONTEXT_OPTIONS = [
    "general",
    "geography",
    "food",
    "daily life",
    "culture",
    "language",
    "history",
    "identity",
    "professions",
]


def normalize_backend_url(url: str) -> str:
    normalized = url.strip().rstrip("/")
    if normalized.endswith("/generate_rag"):
        normalized = normalized[: -len("/generate_rag")]
    elif normalized.endswith("/generate"):
        normalized = normalized[: -len("/generate")]
    elif normalized.endswith("/models"):
        normalized = normalized[: -len("/models")]
    return normalized


def build_models_url(backend_url: str) -> str:
    return normalize_backend_url(backend_url) + "/models"


def build_generate_url(backend_url: str, endpoint: str) -> str:
    return normalize_backend_url(backend_url) + endpoint


def load_default_models() -> list[dict[str, str]]:
    raw = os.getenv("INUKTITUT_MODELS")
    if not raw:
        return DEFAULT_MODELS

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return DEFAULT_MODELS

    if not isinstance(parsed, list):
        return DEFAULT_MODELS

    models: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        label = item.get("label")
        endpoint = item.get("endpoint", "/generate")
        model_type = item.get("model_type", model_id)
        if (
            isinstance(model_id, str)
            and isinstance(label, str)
            and isinstance(endpoint, str)
            and isinstance(model_type, str)
        ):
            models.append(
                {
                    "id": model_id,
                    "label": label,
                    "endpoint": endpoint,
                    "model_type": model_type,
                }
            )
    return models or DEFAULT_MODELS


def build_chat_store_path() -> Path:
    raw_path = os.getenv("INUKTITUT_CHAT_STORE")
    if not raw_path:
        return DEFAULT_CHAT_STORE_PATH
    return Path(raw_path).expanduser()


def call_json_api(url: str, payload: dict[str, Any] | None, timeout_seconds: int) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )

    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Backend returned HTTP {exc.code}: {details or exc.reason}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach the local backend at {url}.") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(
            f"The backend timed out after {timeout_seconds} seconds. Increase the timeout and try again."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Backend response was not valid JSON.") from exc


def fetch_models(config: AppConfig) -> list[dict[str, str]]:
    try:
        body = call_json_api(config.models_url, payload=None, timeout_seconds=config.timeout_seconds)
    except RuntimeError:
        return load_default_models()

    if isinstance(body.get("models"), list):
        normalized_models: list[dict[str, str]] = []
        for model in body["models"]:
            if not isinstance(model, dict):
                continue
            model_id = model.get("id")
            label = model.get("label")
            endpoint = model.get("endpoint", "/generate")
            model_type = model.get("model_type", model_id)
            if (
                isinstance(model_id, str)
                and isinstance(label, str)
                and isinstance(endpoint, str)
                and isinstance(model_type, str)
            ):
                normalized_models.append(
                    {
                        "id": model_id,
                        "label": label,
                        "endpoint": endpoint,
                        "model_type": model_type,
                    }
                )
        return normalized_models or load_default_models()

    models = body.get("models")
    available = body.get("available")
    rag_enabled = body.get("rag")
    if isinstance(available, list):
        normalized_models: list[dict[str, str]] = []
        for model_type in available:
            if not isinstance(model_type, str):
                continue
            label_prefix = "Base" if model_type == "base" else "Adapted"
            normalized_models.append(
                {
                    "id": model_type,
                    "label": f"{label_prefix} model",
                    "endpoint": "/generate",
                    "model_type": model_type,
                }
            )
            if rag_enabled is True:
                normalized_models.append(
                    {
                        "id": f"{model_type}-rag",
                        "label": f"{label_prefix} model + RAG",
                        "endpoint": "/generate_rag",
                        "model_type": model_type,
                    }
                )
        return normalized_models or load_default_models()

    return load_default_models()


def parse_backend_response(body: dict[str, Any], fallback_model_label: str) -> dict[str, Any]:
    answer = body.get("response")
    if not isinstance(answer, str) or not answer.strip():
        answer = body.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        answer = body.get("message")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Backend response did not include a non-empty answer field.")

    sources = body.get("sources", [])
    if not isinstance(sources, list):
        retrieved = body.get("retrieved", [])
        if isinstance(retrieved, list):
            sources = [
                {"instruction": f"Chunk {index + 1}", "context": "retrieved", "response": str(chunk)}
                for index, chunk in enumerate(retrieved)
            ]
        else:
            sources = []

    model_label = body.get("model_label")
    if not isinstance(model_label, str) or not model_label.strip():
        model_label = fallback_model_label

    return {
        "response": answer.strip(),
        "model_label": model_label,
        "sources": sources,
        "raw": body,
    }


def call_local_backend(
    question: str,
    model_config: dict[str, str],
    context_value: str,
    rag_k: int,
    config: AppConfig,
) -> dict[str, Any]:
    endpoint = model_config.get("endpoint", "/generate")
    payload: dict[str, Any] = {
        "question": question,
        "context": context_value,
        "model_type": model_config.get("model_type", "adapted"),
    }
    if endpoint == "/generate_rag":
        payload["k"] = rag_k

    body = call_json_api(
        build_generate_url(config.backend_url, endpoint),
        payload=payload,
        timeout_seconds=config.timeout_seconds,
    )
    return body


def make_chat_title(first_user_message: str) -> str:
    compact = " ".join(first_user_message.split())
    if not compact:
        return "Untitled chat"
    return compact[:40] + ("..." if len(compact) > 40 else "")


def new_chat() -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "title": "Untitled chat",
        "messages": [{"role": "assistant", "content": ASSISTANT_GREETING}],
    }


def is_valid_message(message: Any) -> bool:
    return (
        isinstance(message, dict)
        and isinstance(message.get("role"), str)
        and isinstance(message.get("content"), str)
    )


def normalize_chat(chat: Any) -> dict[str, Any] | None:
    if not isinstance(chat, dict):
        return None

    chat_id = chat.get("id")
    title = chat.get("title")
    messages = chat.get("messages")
    if not isinstance(chat_id, str) or not chat_id.strip():
        return None
    if not isinstance(title, str) or not title.strip():
        title = "Untitled chat"
    if not isinstance(messages, list):
        return None

    normalized_messages = [message for message in messages if is_valid_message(message)]
    if not normalized_messages:
        normalized_messages = [{"role": "assistant", "content": ASSISTANT_GREETING}]

    return {
        "id": chat_id,
        "title": title,
        "messages": normalized_messages,
    }


def load_chats_from_disk(chat_store_path: Path) -> list[dict[str, Any]]:
    if not chat_store_path.exists():
        return []

    try:
        payload = json.loads(chat_store_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(payload, dict):
        return []

    chats = payload.get("chats")
    if not isinstance(chats, list):
        return []

    normalized_chats: list[dict[str, Any]] = []
    for chat in chats:
        normalized = normalize_chat(chat)
        if normalized is not None:
            normalized_chats.append(normalized)
    return normalized_chats


def save_chats_to_disk(chat_store_path: Path) -> None:
    chat_store_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"chats": st.session_state.chats}
    chat_store_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_state(chat_store_path: Path) -> None:
    if "chats" not in st.session_state:
        persisted_chats = load_chats_from_disk(chat_store_path)
        if persisted_chats:
            st.session_state.chats = persisted_chats
            st.session_state.current_chat_id = persisted_chats[0]["id"]
        else:
            first_chat = new_chat()
            st.session_state.chats = [first_chat]
            st.session_state.current_chat_id = first_chat["id"]
            save_chats_to_disk(chat_store_path)


def get_current_chat() -> dict[str, Any]:
    current_chat_id = st.session_state.current_chat_id
    for chat in st.session_state.chats:
        if chat["id"] == current_chat_id:
            return chat

    fallback_chat = new_chat()
    st.session_state.chats = [fallback_chat]
    st.session_state.current_chat_id = fallback_chat["id"]
    return fallback_chat


def render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            model_label = message.get("model_label")
            if isinstance(model_label, str) and model_label:
                st.caption(f"Answered by: {model_label}")

            sources = message.get("sources")
            if isinstance(sources, list) and sources:
                with st.expander("Retrieved context", expanded=False):
                    for source in sources:
                        if not isinstance(source, dict):
                            continue
                        instruction = source.get("instruction", "Unknown")
                        context = source.get("context", "Unknown")
                        response = source.get("response", "")
                        st.markdown(f"**{instruction}**")
                        st.caption(f"Topic: {context}")
                        if response:
                            st.write(response)


chat_store_path = build_chat_store_path()
ensure_state(chat_store_path)

with st.sidebar:
    st.header("Chats")
    if st.button("New chat", use_container_width=True):
        chat = new_chat()
        st.session_state.chats.insert(0, chat)
        st.session_state.current_chat_id = chat["id"]
        save_chats_to_disk(chat_store_path)
        st.rerun()

    current_chat = get_current_chat()
    chat_options = [chat["id"] for chat in st.session_state.chats]
    chat_titles = {chat["id"]: chat["title"] for chat in st.session_state.chats}
    selected_chat_id = st.radio(
        "History",
        options=chat_options,
        index=chat_options.index(current_chat["id"]),
        format_func=lambda chat_id: chat_titles.get(chat_id, "Untitled chat"),
        label_visibility="collapsed",
    )
    st.session_state.current_chat_id = selected_chat_id

    st.divider()
    st.header("Connection")
    backend_url = st.text_input(
        "Backend URL",
        value=os.getenv("INUKTITUT_BACKEND_URL", "http://localhost:8000"),
    )
    models_url = st.text_input(
        "Models URL",
        value=os.getenv("INUKTITUT_MODELS_URL", build_models_url(backend_url)),
    )
    timeout_seconds = st.slider("Timeout", min_value=5, max_value=600, value=120)
    context_value = st.selectbox(
        "Topic",
        options=DEFAULT_CONTEXT_OPTIONS,
        index=DEFAULT_CONTEXT_OPTIONS.index("general"),
    )
    rag_k = st.slider("RAG top-k", min_value=1, max_value=8, value=3)
    config = AppConfig(
        backend_url=normalize_backend_url(backend_url),
        timeout_seconds=timeout_seconds,
        models_url=models_url.strip(),
        chat_store_path=chat_store_path,
    )

    available_models = fetch_models(config)
    model_options = [model["id"] for model in available_models]
    model_labels = {model["id"]: model["label"] for model in available_models}
    model_configs = {model["id"]: model for model in available_models}
    if "selected_model_id" in st.session_state and st.session_state.selected_model_id in model_options:
        default_model_index = model_options.index(st.session_state.selected_model_id)
    elif "adapted-rag" in model_options:
        default_model_index = model_options.index("adapted-rag")
    elif "adapted" in model_options:
        default_model_index = model_options.index("adapted")
    else:
        default_model_index = 0
    selected_model_id = st.selectbox(
        "Model",
        options=model_options,
        index=default_model_index,
        format_func=lambda model_id: model_labels.get(model_id, model_id),
        key="selected_model_id",
    )
    selected_model_label = model_labels.get(selected_model_id, selected_model_id)
    selected_model_config = model_configs[selected_model_id]

    if st.button("Clear current chat", use_container_width=True):
        active_chat = get_current_chat()
        active_chat["title"] = "Untitled chat"
        active_chat["messages"] = [{"role": "assistant", "content": ASSISTANT_GREETING}]
        save_chats_to_disk(config.chat_store_path)
        st.rerun()

current_chat = get_current_chat()

st.title("InuktitutPersonaPlex")

for message in current_chat["messages"]:
    render_message(message)

prompt = st.chat_input("Ask a question")

if prompt:
    user_message = {"role": "user", "content": prompt}
    current_chat["messages"].append(user_message)
    if current_chat["title"] == "Untitled chat":
        current_chat["title"] = make_chat_title(prompt)
    save_chats_to_disk(config.chat_store_path)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                raw_result = call_local_backend(
                    prompt.strip(),
                    selected_model_config,
                    context_value,
                    rag_k,
                    config,
                )
            result = parse_backend_response(raw_result, selected_model_label)
            assistant_message = {
                "role": "assistant",
                "content": result["response"],
                "model_label": result["model_label"],
                "sources": result["sources"],
            }
        except RuntimeError as exc:
            assistant_message = {
                "role": "assistant",
                "content": str(exc),
                "model_label": selected_model_label,
                "sources": [],
            }
        st.markdown(assistant_message["content"])
        if assistant_message["model_label"]:
            st.caption(f"Answered by: {assistant_message['model_label']}")
        if assistant_message["sources"]:
            with st.expander("Retrieved context", expanded=False):
                for source in assistant_message["sources"]:
                    if not isinstance(source, dict):
                        continue
                    instruction = source.get("instruction", "Unknown")
                    context = source.get("context", "Unknown")
                    response = source.get("response", "")
                    st.markdown(f"**{instruction}**")
                    st.caption(f"Topic: {context}")
                    if response:
                        st.write(response)

    current_chat["messages"].append(assistant_message)
    save_chats_to_disk(config.chat_store_path)
