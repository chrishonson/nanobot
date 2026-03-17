import asyncio
import os
import json
from pathlib import Path
from typing import AsyncGenerator
import aiofiles
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

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
async def get_status():
    return {
        "status": "online",
        "workspace": str(config.workspace_path),
        "model": config.agents.defaults.model,
        "version": "0.1.0"
    }

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

        async with aiofiles.open(log_file, mode="r") as f:
            # Go to the end of the file
            await f.seek(0, os.SEEK_END)
            
            while True:
                if await request.is_disconnected():
                    break
                
                line = await f.readline()
                if not line:
                    await asyncio.sleep(0.5)
                    continue
                
                yield f"data: {line.strip()}\n\n"

    return StreamingResponse(log_generator(), media_type="text/event-stream")

# Mount static files (will be created next)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

def start_dashboard(port: int = 18791, host: str = "0.0.0.0"):
    import uvicorn
    uvicorn.run(app, host=host, port=port)
