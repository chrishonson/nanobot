import asyncio
import os
import json
from pathlib import Path
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import webbrowser
import uvicorn
from nanobot.perf.reader import PerfReader

from nanobot.config.loader import load_config
from nanobot.session.manager import SessionManager
from nanobot.config.paths import get_logs_dir

app = FastAPI(title="Nanobot Dashboard")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config and session manager
config = load_config()
session_manager = SessionManager(config.workspace_path)

@app.get("/api/status")
async def get_status(request: Request):
    channels = getattr(request.app.state, "channels", None)
    cron = getattr(request.app.state, "cron", None)
    
    # Read telemetry
    perf_path = get_logs_dir() / "perf.jsonl"
    llm_stats = {}
    try:
        if perf_path.exists():
            reader = PerfReader(perf_path)
            llm_stats = reader.summarize_llm(since_seconds=3600)
    except Exception:
        pass
    
    return {
        "status": "online",
        "workspace": str(config.workspace_path),
        "model": config.agents.defaults.model,
        "version": "0.1.0",
        "channels": list(channels.enabled_channels) if channels else [],
        "cron_jobs": cron.status() if cron else {"jobs": 0, "active": False},
        "perf": llm_stats
    }

class ChatRequest(BaseModel):
    message: str
    session_key: str = "dashboard:main"
    model: str = "cloud"  # Default to cloud model for dashboard

@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    agent = getattr(request.app.state, "agent", None)
    if not agent:
        raise HTTPException(status_code=503, detail="Agent loop not attached to dashboard")

    # Route through @model prefix so the agent loop resolves the named agent
    content = f"@{req.model} {req.message}" if req.model else req.message
        
    queue = asyncio.Queue()

    async def on_stream(delta: str):
        await queue.put({"type": "delta", "content": delta})

    async def on_progress(text: str, tool_hint=False):
        await queue.put({"type": "progress", "content": text, "tool_hint": tool_hint})

    async def on_stream_end(*, resuming=False):
        pass

    async def run_agent():
        try:
            await agent.process_direct(
                content,
                session_key=req.session_key,
                channel="dashboard",
                chat_id="main",
                on_stream=on_stream,
                on_progress=on_progress,
                on_stream_end=on_stream_end
            )
        finally:
            await queue.put(None)

    asyncio.create_task(run_agent())

    async def event_generator():
        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/sessions")
async def list_sessions():
    return session_manager.list_sessions()

@app.get("/api/sessions/{key}")
async def get_session(key: str):
    session = session_manager.get_or_create(key)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "key": session.key,
        "messages": session.messages,
        "metadata": session.metadata,
        "updated_at": session.updated_at.isoformat()
    }

@app.get("/api/logs")
async def stream_logs(request: Request):
    async def log_generator() -> AsyncGenerator[str, None]:
        log_file = get_logs_dir() / "gateway.log"
        if not log_file.exists():
            yield "data: [Error] Log file not found\n\n"
            return

        # Seek to end of file, then tail new lines
        f = open(log_file, "r")
        f.seek(0, os.SEEK_END)
        try:
            while True:
                if await request.is_disconnected():
                    break
                line = f.readline()
                if not line:
                    await asyncio.sleep(0.5)
                    continue
                yield f"data: {line.strip()}\n\n"
        finally:
            f.close()

    return StreamingResponse(log_generator(), media_type="text/event-stream")

# Mount static files (will be created next)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

def start_dashboard(port: int = 18791, host: str = "0.0.0.0"):
    uvicorn.run(app, host=host, port=port)

class DashboardManager:
    def __init__(self, agent, channels, cron, port: int):
        app.state.agent = agent
        app.state.channels = channels
        app.state.cron = cron
        self.port = port
        self.server = None

    async def start(self):
        cfg = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        self.server = uvicorn.Server(cfg)
        
        async def open_browser():
            await asyncio.sleep(1.0)
            webbrowser.open(f"http://127.0.0.1:{self.port}")
            
        asyncio.create_task(open_browser())
        await self.server.serve()

    async def stop(self):
        if self.server:
            self.server.should_exit = True
