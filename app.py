"""
app.py — Streamlit Frontend

A simple chat interface for the Anna University IT Department RAG chatbot.
This is a pure client: it calls the FastAPI backend (main.py) over HTTP —
it does NOT import the pipeline directly. Run the backend separately first.

Usage:
    # Terminal 1 — start the backend
    uvicorn main:app --reload

    # Terminal 2 — start this frontend
    streamlit run app.py
"""

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 60 # seconds — generation can take a few seconds

st.set_page_config(
    page_title="Ask the IT Department",
    page_icon="🎓",
    layout="centered",
)

# ---- Backend health check -------------------------------------------------

def check_backend() -> bool:
    try:
        response = requests.get(f"{API_BASE}/health", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


# ---- Session state ---------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {question, answer, sources, used_search}


# ---- Sidebar ---------------------------------------------------

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "This chatbot answers questions about the **Anna University IT "
        "Department** using its own scraped web content. If the knowledge "
        "base doesn't have enough information, it automatically falls back "
        "to a live web search.\n\n"
        "A Redis-backed **semantic cache** sits in front of the pipeline: "
        "semantically similar questions are answered instantly, with no "
        "LLM call, once they've been asked before."
    )
    st.markdown("---")

    backend_ok = check_backend()
    if backend_ok:
        st.success("Backend connected")
    else:
        st.error("Backend unreachable")
        st.caption("Start it with: `uvicorn main:app --reload`")

    st.markdown("---")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Example questions:")
    example_questions = [
        "What courses does the IT department offer?",
        "How many undergraduate students are in the department?",
        "Who is the Head of the IT department?",
        "What are the research thrust areas?",
    ]
    for eq in example_questions:
        if st.button(eq, key=f"example_{eq}", use_container_width=True):
            st.session_state.pending_question = eq


# ---- Main chat area ---------------------------------------------------

st.title("🎓 Ask the IT Department")
st.caption("Anna University — Department of Information Technology")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message("user"):
        st.write(msg["question"])

    with st.chat_message("assistant"):
        st.write(msg["answer"])

        # Routing badge — shows whether this came from the semantic cache,
        # the KB, or live search
        if msg.get("cache_hit"):
            st.caption("⚡ Answered from semantic cache (no LLM call)")
        elif msg.get("used_search"):
            st.caption("🔎 Answered via live web search (not found in knowledge base)")
        else:
            st.caption("📚 Answered from knowledge base")

        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"{i}. [{src}]({src})")

        if msg.get("response_time_seconds") is not None:
            st.caption(f"⏱ {msg['response_time_seconds']}s")


def ask_question(question: str):
    """Call the backend and append the result to chat history."""
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Routing your question…"):
            try:
                response = requests.post(
                    f"{API_BASE}/query",
                    json={"question": question},
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()

                st.write(data["answer"])

                if data.get("cache_hit"):
                    st.caption("⚡ Answered from semantic cache (no LLM call)")
                elif data.get("used_search"):
                    st.caption("🔎 Answered via live web search (not found in knowledge base)")
                else:
                    st.caption("📚 Answered from knowledge base")

                if data.get("sources"):
                    with st.expander(f"Sources ({len(data['sources'])})"):
                        for i, src in enumerate(data["sources"], 1):
                            st.markdown(f"{i}. [{src}]({src})")

                st.caption(f"⏱ {data.get('response_time_seconds', '?')}s")

                st.session_state.messages.append({
                    "question": question,
                    "answer": data["answer"],
                    "sources": data.get("sources", []),
                    "used_search": data.get("used_search", False),
                    "cache_hit": data.get("cache_hit", False),
                    "response_time_seconds": data.get("response_time_seconds"),
                })

            except requests.exceptions.ConnectionError:
                error_msg = (
                    "Can't reach the backend. Make sure it's running: "
                    "`uvicorn main:app --reload`"
                )
                st.error(error_msg)
                st.session_state.messages.append({
                    "question": question, "answer": error_msg,
                    "sources": [], "used_search": False, "response_time_seconds": None,
                })

            except requests.exceptions.Timeout:
                error_msg = "The request timed out. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({
                    "question": question, "answer": error_msg,
                    "sources": [], "used_search": False, "response_time_seconds": None,
                })

            except requests.exceptions.HTTPError as e:
                error_msg = f"Server error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "question": question, "answer": error_msg,
                    "sources": [], "used_search": False, "response_time_seconds": None,
                })


# Handle example-question button clicks from the sidebar
if "pending_question" in st.session_state:
    q = st.session_state.pop("pending_question")
    ask_question(q)
    st.rerun()

# Chat input box
if question := st.chat_input("Ask a question about the IT department..."):
    if not backend_ok:
        st.error("Backend is not reachable. Start it first with `uvicorn main:app --reload`.")
    else:
        ask_question(question)
        st.rerun()
