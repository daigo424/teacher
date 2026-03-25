import json
import os
from typing import Any

import requests
import streamlit as st

st.set_page_config(
    page_title="Teacher API Playground",
    page_icon="📚",
    layout="wide",
)

def load_api_base_url() -> str:
    env_value = os.getenv("API_BASE_URL")
    if env_value:
        return env_value.rstrip("/")

    try:
        secret_value = st.secrets["API_BASE_URL"]
        if secret_value:
            return str(secret_value).rstrip("/")
    except Exception:
        pass

    return "http://localhost:8000"

DEFAULT_API_BASE_URL = load_api_base_url()

def get_api_base_url() -> str:
    return st.session_state.get("api_base_url", DEFAULT_API_BASE_URL).rstrip("/")


def call_health(api_base_url: str) -> tuple[bool, str]:
    try:
        res = requests.get(f"{api_base_url}/health", timeout=10)
        res.raise_for_status()
        return True, res.text
    except requests.RequestException as e:
        return False, str(e)


def call_ask(api_base_url: str, question: str) -> dict[str, Any]:
    res = requests.post(
        f"{api_base_url}/ask",
        json={"question": question},
        timeout=60,
    )
    res.raise_for_status()
    return res.json()


def render_sources(sources: Any) -> None:
    if not sources:
        st.info("No sources were returned.")
        return

    if isinstance(sources, list):
        for i, item in enumerate(sources, start=1):
            with st.expander(f"Source {i}", expanded=False):
                if isinstance(item, dict):
                    st.json(item)
                else:
                    st.write(item)
    else:
        st.json(sources)


st.title("📚 Teacher API Playground")
st.caption("A demo app to test the FastAPI /ask endpoint via Streamlit")

with st.sidebar:
    st.header("Settings")
    st.text_input(
        "API Base URL",
        value=DEFAULT_API_BASE_URL,
        key="api_base_url",
        help="e.g. http://localhost:8000 or https://your-api.onrender.com",
    )

    api_base_url = get_api_base_url()

    if st.button("Health Check"):
        ok, msg = call_health(api_base_url)
        if ok:
            st.success(f"API is healthy: {msg}")
        else:
            st.error(f"Health check failed: {msg}")

    st.markdown("---")
    st.markdown("### Example Questions")
    example_questions = [
        "What is artificial intelligence?",
        "What is machine learning?",
        "How is deep learning different from machine learning?",
        "What can this system do?",
    ]

    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state["question"] = q

question = st.text_area(
    "Question",
    value=st.session_state.get("question", ""),
    height=140,
    placeholder="Ask something...",
)

col1, col2 = st.columns([1, 1])

with col1:
    ask_clicked = st.button("Ask", type="primary", use_container_width=True)

with col2:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state["question"] = ""
    st.rerun()

if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    api_base_url = get_api_base_url()

    with st.spinner("Calling /ask ..."):
        try:
            data = call_ask(api_base_url, question.strip())
        except requests.HTTPError as e:
            body = e.response.text if e.response is not None else str(e)
            st.error("The API returned an error.")
            st.code(body)
            st.stop()
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    st.subheader("Answer")
    answer = data.get("answer", "(no answer)")
    st.write(answer)

    meta_cols = st.columns(2)
    with meta_cols[0]:
        st.metric("context_count", data.get("context_count", 0))
    with meta_cols[1]:
        sources = data.get("sources")
        source_count = len(sources) if isinstance(sources, list) else 0
        st.metric("sources", source_count)

    st.subheader("Sources")
    render_sources(data.get("sources"))

    with st.expander("Raw Response JSON"):
        st.code(json.dumps(data, ensure_ascii=False, indent=2), language="json")