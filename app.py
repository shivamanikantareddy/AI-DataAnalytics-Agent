"""
app.py — Streamlit frontend for the AI-assisted Data Analytics platform.

Stages:
  upload      → user picks a CSV/Excel/JSON file
  processing  → (implicit) /upload_and_start blocks; spinner shown while waiting
  chatting    → interrupt() fired; chat UI rendered
  visualizing → user typed "done"; polling until stage == done
  done        → visualization gallery rendered

Run:
  streamlit run app.py
"""

import time
import requests
import streamlit as st

# ── Config ───────────────────────────────────────────────────────────────────

API_BASE    = "http://localhost:8000"   # Change if FastAPI runs elsewhere
POLL_INTERVAL = 2                       # seconds between /status polls

# ── Page setup ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Data Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI-assisted Data Analytics")
st.caption("Upload a dataset, explore it with AI, then generate visualizations.")

# ── Session state defaults ───────────────────────────────────────────────────

def _init_state():
    defaults = {
        "thread_id":    None,
        "stage":        "upload",       # upload | chatting | visualizing | done
        "chat_history": [],             # list of {"role": "user"|"assistant", "content": str}
        "viz_images":   [],             # list of image URLs
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()


# ── Helper: API calls ────────────────────────────────────────────────────────

def api_upload(file_bytes: bytes, filename: str) -> dict:
    """POST /upload_and_start — blocks until interrupt() fires."""
    resp = requests.post(
        f"{API_BASE}/upload_and_start",
        files={"file": (filename, file_bytes)},
        timeout=900,   # pipeline can take minutes
    )
    resp.raise_for_status()
    return resp.json()


def api_status(thread_id: str) -> dict:
    resp = requests.get(f"{API_BASE}/status/{thread_id}", timeout=15)
    resp.raise_for_status()
    return resp.json()


def api_chat(thread_id: str, message: str) -> dict:
    resp = requests.post(
        f"{API_BASE}/chat/{thread_id}",
        json={"message": message},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def api_visualizations(thread_id: str) -> list[str]:
    resp = requests.get(f"{API_BASE}/visualizations/{thread_id}", timeout=30)
    resp.raise_for_status()
    return resp.json().get("images", [])


def api_delete_session(thread_id: str):
    try:
        requests.delete(f"{API_BASE}/session/{thread_id}", timeout=10)
    except Exception:
        pass   # Best-effort cleanup


def poll_until(thread_id: str, target_stage: str, spinner_msg: str):
    """
    Polls /status every POLL_INTERVAL seconds until stage matches target_stage.
    Shows a Streamlit spinner while waiting.
    Returns the final status dict.
    """
    with st.spinner(spinner_msg):
        while True:
            status = api_status(thread_id)
            if status["stage"] == target_stage:
                return status
            if status["stage"] == "unknown":
                st.error("Session expired or server restarted. Please start over.")
                _reset()
                st.stop()
            time.sleep(POLL_INTERVAL)
            st.rerun()


def _reset():
    """Clear all session state and go back to the upload screen."""
    if st.session_state.thread_id:
        api_delete_session(st.session_state.thread_id)
    for key in ["thread_id", "stage", "chat_history", "viz_images"]:
        del st.session_state[key]
    _init_state()


# ── Stage: upload ────────────────────────────────────────────────────────────

if st.session_state.stage == "upload":

    uploaded = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx", "xls", "json"],
        help="Supported formats: CSV, Excel, JSON. Max 500 MB.",
    )

    if uploaded is not None:
        st.info(f"**{uploaded.name}** selected — click below to start analysis.")
        if st.button("🚀 Start Analysis", type="primary"):
            with st.spinner(
                "Uploading and running pipeline… "
                "Cleaning → EDA → Analysis. This may take a few minutes."
            ):
                try:
                    result = api_upload(uploaded.read(), uploaded.name)
                except requests.HTTPError as e:
                    st.error(f"Server error: {e.response.text}")
                    st.stop()
                except requests.ConnectionError:
                    st.error("Cannot reach the backend. Is FastAPI running?")
                    st.stop()

            st.session_state.thread_id = result["thread_id"]
            st.session_state.stage     = "chatting"

            # Show the first AI message that was waiting at interrupt()
            first_msg = result.get("latest_message", "")
            if first_msg:
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": first_msg}
                )
            st.rerun()


# ── Stage: chatting ──────────────────────────────────────────────────────────

elif st.session_state.stage == "chatting":

    st.success("✅ Analysis complete — ask questions about your data below.")
    st.caption('Type your questions. When you\'re done, type **"done"** to generate visualizations.')

    # Render chat history
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about your dataset…")

    if user_input:
        # Append user message immediately
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Send to backend
        with st.spinner("Thinking…"):
            try:
                result = api_chat(st.session_state.thread_id, user_input)
            except requests.HTTPError as e:
                st.error(f"Chat error: {e.response.text}")
                st.stop()

        ai_reply = result.get("latest_message", "")
        new_stage = result.get("stage", "chatting")

        if ai_reply:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": ai_reply}
            )

        if new_stage in ("visualizing", "done"):
            st.session_state.stage = "visualizing"
        # else stays "chatting"

        st.rerun()

    # Start Over button
    with st.sidebar:
        if st.button("🔄 Start Over"):
            _reset()
            st.rerun()


# ── Stage: visualizing ───────────────────────────────────────────────────────

elif st.session_state.stage == "visualizing":

    st.info("Generating visualizations… please wait.")

    # Poll until stage is 'done'
    final_status = poll_until(
        thread_id=st.session_state.thread_id,
        target_stage="done",
        spinner_msg="Generating charts and graphs…",
    )

    # Fetch image URLs
    try:
        images = api_visualizations(st.session_state.thread_id)
    except requests.HTTPError as e:
        st.error(f"Could not fetch visualizations: {e.response.text}")
        images = []

    st.session_state.viz_images = images
    st.session_state.stage = "done"
    st.rerun()


# ── Stage: done ──────────────────────────────────────────────────────────────

elif st.session_state.stage == "done":

    st.success("✅ Analysis and visualizations complete!")

    images = st.session_state.viz_images

    if not images:
        st.warning("No visualizations were generated.")
    else:
        st.subheader(f"📈 Generated Visualizations ({len(images)})")

        # Responsive 2-column grid
        cols = st.columns(2)
        for idx, img_url in enumerate(images):
            full_url = f"{API_BASE}{img_url}"
            with cols[idx % 2]:
                st.image(full_url, use_container_width=True)

    # Show chat history summary in expander
    if st.session_state.chat_history:
        with st.expander("💬 Chat history"):
            for turn in st.session_state.chat_history:
                label = "**You:**" if turn["role"] == "user" else "**Assistant:**"
                st.markdown(f"{label} {turn['content']}")

    with st.sidebar:
        if st.button("🔄 Start Over", type="primary"):
            _reset()
            st.rerun()
