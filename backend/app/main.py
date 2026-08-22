import asyncio
import base64
import json
import logging
import uuid
import time
from typing import Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.pipeline.graph import run_pipeline
from app.pipeline.schemas import PipelineInput, PipelineOutput, HealthResponse
from app.retrieval.qdrant_client import get_qdrant
from app.retrieval.embedder import get_embedder
from app.guardrails.grounding import _load_nli_model
from app.utils.audio import base64_to_pcm, pcm_to_base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Session storage
sessions: Dict[str, dict] = {}

# Text-only test input model
class TextQueryInput(BaseModel):
    query: str
    language: str = "en"
    session_id: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Voice RAG API...")
    qdrant = get_qdrant()
    try:
        # Test connection - just check a few collections
        for lang in settings.supported_languages[:3]:
            qdrant.get_collection_stats(lang)
        logger.info("Qdrant connection verified")
    except Exception as e:
        logger.warning(f"Qdrant connection issue: {e}")
    
    # Pre-load models in background to avoid first-request latency
    logger.info("Pre-loading embedder...")
    try:
        get_embedder()
        logger.info("Embedder loaded")
    except Exception as e:
        logger.warning(f"Embedder pre-load failed: {e}")
    
    logger.info("Pre-loading NLI model...")
    try:
        _load_nli_model()
        logger.info("NLI model loaded")
    except Exception as e:
        logger.warning(f"NLI model pre-load failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Voice RAG API",
    description="Voice-enabled RAG for MSMARCO-XI (18 Indic languages)",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"Client connected: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"Client disconnected: {session_id}")
    
    async def send_json(self, session_id: str, data: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(data)
            except Exception as e:
                logger.error(f"Send failed for {session_id}: {e}")


manager = ConnectionManager()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    qdrant = get_qdrant()
    collections = {}
    sarvam_ok = bool(settings.sarvam_api_key)
    qdrant_ok = False
    
    try:
        for lang in settings.supported_languages[:3]:
            stats = qdrant.get_collection_stats(lang)
            collections[lang] = stats.get("points_count", 0)
        qdrant_ok = True
    except:
        pass
    
    return HealthResponse(
        status="healthy" if (sarvam_ok and qdrant_ok) else "degraded",
        sarvam_connected=sarvam_ok,
        qdrant_connected=qdrant_ok,
        collections=collections
    )


@app.post("/api/query", response_model=PipelineOutput)
async def query_endpoint(input_data: PipelineInput):
    """REST endpoint for voice query."""
    session_id = input_data.session_id or str(uuid.uuid4())
    
    # Decode audio
    audio_bytes = base64_to_pcm(input_data.audio_base64)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Invalid audio data")
    
    # Get conversation history
    history = sessions.get(session_id, {}).get("history", [])
    
    # Run pipeline
    result = await run_pipeline(
        audio_bytes=audio_bytes,
        language=input_data.language,
        session_id=session_id,
        conversation_history=history
    )
    
    # Update session
    if session_id not in sessions:
        sessions[session_id] = {}
    sessions[session_id]["history"] = result.get("conversation_history", [])
    
    # Encode response audio
    audio_b64 = pcm_to_base64(result.get("tts_audio", b""))
    
    return PipelineOutput(
        transcript=result.get("transcript", ""),
        answer=result.get("answer", ""),
        audio_base64=audio_b64,
        language=result.get("detected_language", input_data.language),
        grounded=result.get("grounded", False),
        refusal_reason=result.get("refusal_reason"),
        latency_ms=result.get("total_latency_ms", 0),
        trace_id=result.get("trace_id", "")
    )


@app.post("/api/query/text", response_model=PipelineOutput)
async def text_query_endpoint(input_data: TextQueryInput):
    """REST endpoint for text-only query (bypasses STT for testing)."""
    session_id = input_data.session_id or str(uuid.uuid4())
    
    # Get conversation history
    history = sessions.get(session_id, {}).get("history", [])
    
    # Create mock audio (will be ignored since we inject transcript directly)
    mock_audio = b"\x00" * 100  # Minimal mock
    
    # Run pipeline with injected transcript
    result = await run_pipeline(
        audio_bytes=mock_audio,
        language=input_data.language,
        session_id=session_id,
        conversation_history=history,
        transcript=input_data.query  # Pass pre-transcribed query
    )
    
    # Update session
    if session_id not in sessions:
        sessions[session_id] = {}
    sessions[session_id]["history"] = result.get("conversation_history", [])
    
    # Encode response audio
    audio_b64 = pcm_to_base64(result.get("tts_audio", b""))
    
    return PipelineOutput(
        transcript=input_data.query,
        answer=result.get("answer", ""),
        audio_base64=audio_b64,
        language=result.get("detected_language", input_data.language),
        grounded=result.get("grounded", False),
        refusal_reason=result.get("refusal_reason"),
        latency_ms=result.get("total_latency_ms", 0),
        trace_id=result.get("trace_id", "")
    )


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time voice interaction."""
    await manager.connect(websocket, session_id)
    
    # Initialize session
    if session_id not in sessions:
        sessions[session_id] = {"history": [], "language": "en"}
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "audio":
                # Process audio chunk
                audio_b64 = data.get("audio_base64", "")
                language = data.get("language", sessions[session_id].get("language", "en"))
                
                audio_bytes = base64_to_pcm(audio_b64)
                if not audio_bytes:
                    await manager.send_json(session_id, {
                        "type": "error",
                        "message": "Invalid audio data"
                    })
                    continue
                
                # Send processing status
                await manager.send_json(session_id, {"type": "processing", "status": "stt"})
                
                # Run pipeline
                history = sessions[session_id].get("history", [])
                result = await run_pipeline(
                    audio_bytes=audio_bytes,
                    language=language,
                    session_id=session_id,
                    conversation_history=history
                )
                
                # Update session
                sessions[session_id]["history"] = result.get("conversation_history", [])
                sessions[session_id]["language"] = result.get("detected_language", language)
                
                # Send results
                audio_b64_out = pcm_to_base64(result.get("tts_audio", b""))
                
                await manager.send_json(session_id, {
                    "type": "result",
                    "transcript": result.get("transcript", ""),
                    "answer": result.get("answer", ""),
                    "audio_base64": audio_b64_out,
                    "language": result.get("detected_language", language),
                    "grounded": result.get("grounded", False),
                    "refusal_reason": result.get("refusal_reason"),
                    "latency_ms": result.get("total_latency_ms", 0),
                    "trace_id": result.get("trace_id", "")
                })
                
            elif msg_type == "config":
                # Update session config
                if "language" in data:
                    sessions[session_id]["language"] = data["language"]
                await manager.send_json(session_id, {"type": "config_ack", "status": "ok"})
                
            elif msg_type == "ping":
                await manager.send_json(session_id, {"type": "pong"})
                
            elif msg_type == "clear_history":
                sessions[session_id]["history"] = []
                await manager.send_json(session_id, {"type": "history_cleared"})
                
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(session_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )