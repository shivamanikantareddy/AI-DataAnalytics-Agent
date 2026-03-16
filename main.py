from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import uuid
import time

from graph import start_graph, get_graph_status, send_chat_message, get_visualization_paths

app = FastAPI(title="AI-assisted Data Analysis and Visualisation API", version="2.0.0")

UPLOAD_DIR = Path("./uploads")
VIZ_DIR    = Path("./visualizations")
UPLOAD_DIR.mkdir(exist_ok=True)
VIZ_DIR.mkdir(exist_ok=True)

# Serve visualization images as static files so Streamlit can display them via URL
app.mount("/images", StaticFiles(directory=str(VIZ_DIR)), name="images")


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


# ── Internal helpers ─────────────────────────────────────────────────────────

def _wait_for_stage(
    thread_id: str,
    target_stages: set[str],
    max_retries: int = 60,
    interval: float = 1.0,
    require_message: bool = False,      # ← new param
    last_user_message: str = "",        # ← new param
) -> dict:
    for attempt in range(max_retries):
        status = get_graph_status(thread_id)
        if status["stage"] in target_stages:
            # If we need a fresh AI reply, keep waiting until it appears
            if require_message:
                msg = status.get("latest_message", "")
                if not msg or msg.strip() == last_user_message.strip():
                    time.sleep(interval)
                    continue
            return status
        if status["stage"] == "unknown":
            raise RuntimeError(f"Thread '{thread_id}' not found.")
        time.sleep(interval)

    status = get_graph_status(thread_id)
    if status["stage"] in target_stages:
        return status
    raise RuntimeError(
        f"Graph did not reach {target_stages} after {max_retries}s. "
        f"Last stage: {status['stage']}"
    )


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Analytics & Visualisation Assistant", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "version": "2.0.0"}


@app.post("/upload_and_start")
async def upload_and_start(
    file: UploadFile = File(..., description="CSV, Excel, or JSON dataset")
):
    """
    1. Saves the uploaded file to ./uploads/{thread_id}/
    2. Starts graph execution (blocks until interrupt() at Chat_node or END).
    3. Polls until the checkpointer confirms the graph is at 'chatting'.
    4. Returns thread_id and the first AI message to the client.
    """
    MAX_SIZE = 500 * 1024 * 1024  # 500 MB

    if file.size and file.size > MAX_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 500 MB limit.")

    thread_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / thread_id
    session_dir.mkdir(parents=True, exist_ok=True)

    file_path = session_dir / file.filename
    contents = await file.read()
    file_path.write_bytes(contents)

    # ── Run the graph (blocks until interrupt() or END) ──────────────────────
    try:
        start_graph(file_path=file_path, thread_id=thread_id)
    except RuntimeError as e:
        import shutil
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

    # ── Wait for checkpointer to confirm the stage ───────────────────────────
    # start_graph() blocks until interrupt(), but the SQLite checkpointer
    # may not have flushed the final state yet when control returns here.
    # Polling with _wait_for_stage() handles that race condition.
    try:
        status = _wait_for_stage(
            thread_id=thread_id,
            target_stages={"chatting", "done"},
            max_retries=30,
            interval=1.0,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(
        status_code=200,
        content={
            "thread_id":      thread_id,
            "stage":          status["stage"],
            "latest_message": status.get("latest_message", ""),
        },
    )


@app.get("/status/{thread_id}")
def get_status(thread_id: str):
    """
    Poll the current stage of a graph thread.

    Returns:
      {
        "stage":          "chatting" | "visualizing" | "done" | "unknown",
        "latest_message": "<most recent AI message>"
      }
    """
    status = get_graph_status(thread_id)
    if status["stage"] == "unknown":
        raise HTTPException(status_code=404, detail="Session not found.")
    return JSONResponse(status_code=200, content=status)


@app.post("/chat/{thread_id}")
def chat(thread_id: str, body: ChatRequest):
    status = get_graph_status(thread_id)
    if status["stage"] == "unknown":
        raise HTTPException(status_code=404, detail="Session not found.")
    if status["stage"] != "chatting":
        raise HTTPException(
            status_code=409,
            detail=f"Graph is not waiting for input (current stage: {status['stage']})."
        )

    try:
        send_chat_message(thread_id=thread_id, user_message=body.message)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Wait for graph to settle back at interrupt() or finish
    try:
        result = _wait_for_stage(
        thread_id=thread_id,
        target_stages={"chatting", "visualizing", "done"},
        max_retries=60,
        interval=1.0,
        require_message=True,               # ← wait for actual AI reply
        last_user_message=body.message,     # ← so we don't return early
    )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(status_code=200, content=result)


@app.get("/visualizations/{thread_id}")
def list_visualizations(thread_id: str):
    """
    Returns a list of image URLs the Streamlit frontend can pass to st.image().
    Call this only after /status returns stage='done'.
    """
    status = get_graph_status(thread_id)
    if status["stage"] == "unknown":
        raise HTTPException(status_code=404, detail="Session not found.")
    if status["stage"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Visualizations not ready yet (stage: {status['stage']})."
        )

    raw_paths = get_visualization_paths(thread_id)

    image_urls = []
    for p in raw_paths:
        filename = Path(p).name
        image_urls.append(f"/images/{filename}")

    return JSONResponse(status_code=200, content={"images": image_urls})


@app.delete("/session/{thread_id}")
def delete_session(thread_id: str):
    """
    Purges uploaded files for this session.
    Call this when the user clicks 'Start Over' in Streamlit.
    """
    import shutil
    session_dir = UPLOAD_DIR / thread_id
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
        
    vis_dir = VIZ_DIR
    if vis_dir.exists():
        shutil.rmtree(vis_dir,ignore_errors=True)

    return JSONResponse(
        status_code=200,
        content={"message": f"Session {thread_id} deleted."}
    )