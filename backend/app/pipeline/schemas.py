from typing import TypedDict, List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class Chunk(BaseModel):
    id: str
    content: str
    language: str
    query_id: int
    query_type: str
    is_selected: bool
    score: float
    source: str
    english_content: str
    metadata: Dict[str, Any] = {}


class ConversationTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    language: str = "en"


class PipelineState(TypedDict):
    # Input
    audio_bytes: bytes
    language: str
    session_id: str
    file_path: str  # Document file for Vision OCR ingestion
    
    # STT
    transcript: str
    stt_confidence: float
    stt_latency_ms: float
    
    # Language detection
    detected_language: str
    lang_confidence: float
    
    # Retrieval
    query_embedding: List[float]
    retrieved_chunks: List[Chunk]
    reranked_chunks: List[Chunk]
    retrieval_latency_ms: float
    
    # Guardrails
    input_guard_passed: bool
    input_guard_reason: Optional[str]
    retrieval_confidence: float
    
    # Generation
    answer: str
    generation_latency_ms: float
    reasoning_content: Optional[str]
    use_mock_generation: bool  # Use mock generation for evaluation
    
    # Verification
    grounded: bool
    grounding_score: float
    hallucination_check_passed: bool
    hallucination_score: float
    
    # Output guard
    output_guard_passed: bool
    output_guard_reason: Optional[str]
    
    # Refusal
    refusal_reason: Optional[str]
    refusal_message: str
    
    # TTS
    tts_audio: bytes
    tts_latency_ms: float
    
    # Conversation
    conversation_history: List[ConversationTurn]
    
    # Document ingestion (Vision OCR)
    file_path: str
    extracted_text: str
    vision_job_id: str
    vision_output_url: str
    ingested_chunks: int
    ingestion_language: str
    vision_error: str
    
    # Metrics
    total_latency_ms: float
    trace_id: str
    component_latencies: Dict[str, float]


class PipelineInput(BaseModel):
    audio_base64: str
    language: str = "en"
    session_id: Optional[str] = None


class PipelineOutput(BaseModel):
    transcript: str
    answer: str
    audio_base64: str
    language: str
    grounded: bool
    refusal_reason: Optional[str]
    latency_ms: float
    trace_id: str


class HealthResponse(BaseModel):
    status: str
    sarvam_connected: bool
    qdrant_connected: bool
    collections: Dict[str, int]
    timestamp: datetime = Field(default_factory=datetime.now)