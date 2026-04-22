import json
import os
from dataclasses import dataclass
from urllib import error, request

import streamlit as st  # type: ignore[reportMissingImports]


st.set_page_config(page_title="InuktitutPersonaPlex", page_icon="ᐃ", layout="centered")


@dataclass(frozen=True)
class AppConfig:
    api_url: str
    timeout_seconds: int


ASSISTANT_GREETING = (
    "Ask me something about Inuktitut language, Inuit communities, food, culture, or history."
)


def call_local_backend(question: str, config: AppConfig) -> str:
    payload = json.dumps({"question": question, "model": "Fine-tuned model"}).encode("utf-8")
    req = request.Request(
        config.api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=config.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Backend returned HTTP {exc.code}: {details or exc.reason}") from exc
    except error.URLError as exc:
        raise RuntimeError(
            f"Could not reach the local backend at {config.api_url}. Make sure the model server is running."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Backend response was not valid JSON.") from exc

    answer = body.get("response")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Backend response did not include a non-empty 'response' field.")
    return answer.strip()


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": ASSISTANT_GREETING}]


with st.sidebar:
    st.header("Connection")
    api_url = st.text_input(
        "API URL",
        value=os.getenv("INUKTITUT_API_URL", "http://localhost:8000/generate"),
    )
    timeout_seconds = st.slider("Timeout", min_value=5, max_value=120, value=30)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": ASSISTANT_GREETING}]
        st.rerun()

config = AppConfig(
    api_url=api_url.strip(),
    timeout_seconds=timeout_seconds,
)

st.title("InuktitutPersonaPlex")
st.caption("Fine-tuned Inuktitut chatbot")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                answer = call_local_backend(prompt.strip(), config)
        except RuntimeError as exc:
            answer = str(exc)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
