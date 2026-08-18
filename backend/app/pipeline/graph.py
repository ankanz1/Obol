import time
import base64
import logging
import uuid
from typing import Dict, Any, List
import asyncio

from langgraph.graph import StateGraph, START, END

from app.config import settings
from app.pipeline.schemas import PipelineState, Chunk, ConversationTurn
from app.retrieval.embedder import get_embedder
from app.retrieval.qdrant_client import get_qdrant
from app.guardrails.input_guard import check_input_safety
from app.guardrails.grounding import check_grounding
from app.guardrails.hallucination import check_hallucination
from app.guardrails.output_guard import check_output_safety
from app.generation.sarvam_client import generate_answer
from app.stt.sarvam_client import transcribe_audio
from app.tts.sarvam_client import synthesize_speech
from app.utils.language import detect_language
from app.mcp import MCPFallback
from data.chunking.strategies import SlidingWindowChunker

logger = logging.getLogger(__name__)


def time_it(func):
    """Decorator to measure execution time."""
    async def wrapper(state: PipelineState, *args, **kwargs):
        start = time.perf_counter()
        result = await func(state, *args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        return result, elapsed
    return wrapper


# ===== NODES =====

async def vision_extract_node(state: PipelineState) -> PipelineState:
    """Extract text from uploaded document/image using Sarvam Vision."""
    file_path = state.get("file_path")
    if not file_path:
        return state
    
    try:
        # Submit job and poll for completion
        from app.mcp.vision_direct import vision_submit_job, vision_poll_status
        
        # Convert language code to BCP-47 format (e.g., "hi" -> "hi-IN")
        lang = state.get("language", "hi")
        lang_code = f"{lang}-IN" if len(lang) == 2 and lang != "en" else lang
        
        job = await vision_submit_job(file_path, "md", lang_code)
        job_id = job["job_id"]
        state["vision_job_id"] = job_id
        
        result = await vision_poll_status(job_id)
        state["vision_output_url"] = result.get("output_url", "")
        
        # Note: Full text extraction requires MCP server for SAS token download
        # The job completed successfully - extracted text would be in the output ZIP
        state["extracted_text"] = f"[Vision OCR job {job_id} completed successfully. Output in {result.get('job_details', [{}])[0].get('outputs', [{}])[0].get('file_name', 'document.zip')}. Use MCP server for text extraction.]"
        
        logger.info(f"Vision job {job_id} completed successfully")
    except Exception as e:
        logger.error(f"Vision extraction failed: {e}")
        import traceback
        traceback.print_exc()
        state["extracted_text"] = ""
        state["vision_error"] = str(e)
    
    return state


async def document_ingest_node(state: PipelineState) -> PipelineState:
    """Ingest extracted document text into Qdrant index."""
    extracted_text = state.get("extracted_text", "")
    if not extracted_text:
        return state
    
    try:
        # Chunk the extracted text using sliding window
        chunker = SlidingWindowChunker(
            window_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        chunks = chunker._split_text(extracted_text)
        
        if not chunks:
            return state
        
        # Convert to chunk dicts
        chunk_dicts = [
            {"id": str(uuid.uuid4()), "content": c}
            for c in chunks
        ]
        
        # Embed chunks
        embedder = get_embedder()
        chunk_texts = [c["content"] for c in chunk_dicts]
        embeddings = embedder.embed(chunk_texts)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunk_dicts):
            chunk["vector"] = embeddings[i]
            # Determine language - use detected_language if available, otherwise initial language
            detected_lang = state.get("detected_language", "")
            initial_lang = state.get("language", "en")
            lang = detected_lang if detected_lang else initial_lang
            if not lang:
                lang = "en"
            chunk["language"] = lang
            chunk["source"] = "vision_extract"
            chunk["file_path"] = state.get("file_path", "")
        
        # Upsert to Qdrant
        qdrant = get_qdrant()
        detected_lang = state.get("detected_language", "")
        initial_lang = state.get("language", "en")
        lang = detected_lang if detected_lang else initial_lang
        if not lang:
            lang = "en"
        qdrant.upsert_chunks(chunk_dicts, lang)
        
        state["ingested_chunks"] = len(chunk_dicts)
        state["ingestion_language"] = lang
        logger.info(f"Ingested {len(chunk_dicts)} chunks from vision extraction into {lang}")
        
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")
        state["ingested_chunks"] = 0
        state["ingestion_error"] = str(e)
    
    return state


async def stt_node(state: PipelineState) -> PipelineState:
    """Speech-to-text using Sarvam Saaras v3 realtime."""
    trace_id = state.get("trace_id", str(uuid.uuid4())[:8])
    state["trace_id"] = trace_id
    state["component_latencies"] = {}
    
    # If transcript already provided (text-only mode), skip STT
    if state.get("transcript"):
        state["stt_confidence"] = 1.0
        state["component_latencies"]["stt"] = 0
        return state
    
    try:
        transcript, confidence = await transcribe_audio(
            state["audio_bytes"],
            state.get("language", "en")
        )
        state["transcript"] = transcript
        state["stt_confidence"] = confidence
        state["component_latencies"]["stt"] = 0  # Will be set by caller
    except Exception as e:
        logger.error(f"STT failed: {e}")
        state["transcript"] = ""
        state["stt_confidence"] = 0.0
        state["refusal_reason"] = "stt_failed"
    
    return state


async def lang_detect_node(state: PipelineState) -> PipelineState:
    """Detect language from transcript."""
    if not state.get("transcript"):
        state["detected_language"] = state.get("language", "en")
        state["lang_confidence"] = 1.0
        return state
    
    try:
        lang, confidence = detect_language(state["transcript"])
        state["detected_language"] = lang
        state["lang_confidence"] = confidence
        
        # Validate supported language
        if lang not in settings.supported_languages:
            state["refusal_reason"] = "unsupported_language"
            state["refusal_message"] = settings.refusal_messages.get("en", "Unsupported language")
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        state["detected_language"] = state.get("language", "en")
        state["lang_confidence"] = 0.5
    
    return state


async def embed_node(state: PipelineState) -> PipelineState:
    """Generate query embedding."""
    if not state.get("transcript"):
        state["query_embedding"] = [0.0] * settings.embedding_dim
        return state
    
    embedder = get_embedder()
    embedding = embedder.embed_single(state["transcript"])
    state["query_embedding"] = embedding.tolist()
    return state


async def retrieve_node(state: PipelineState) -> PipelineState:
    """Retrieve relevant chunks from Qdrant."""
    if not state.get("transcript"):
        state["retrieved_chunks"] = []
        return state
    
    qdrant = get_qdrant()
    lang = state.get("detected_language", state.get("language", "en"))
    
    try:
        results = qdrant.search(
            query_vector=state["query_embedding"],
            language=lang,
            limit=settings.retrieval_limit,
            score_threshold=settings.retrieval_score_threshold
        )
        
        chunks = []
        for r in results:
            payload = r.payload
            chunks.append(Chunk(
                id=str(r.id),
                content=payload.get("content", ""),
                language=payload.get("language", lang),
                query_id=payload.get("query_id", 0),
                query_type=payload.get("query_type", "UNKNOWN"),
                is_selected=payload.get("is_selected", False),
                score=r.score,
                source=payload.get("source", "unknown"),
                english_content=payload.get("english_content", ""),
                metadata={k: v for k, v in payload.items() if k not in [
                    "content", "language", "query_id", "query_type", "is_selected", "source", "english_content"
                ]}
            ))
        
        state["retrieved_chunks"] = chunks
        state["retrieval_confidence"] = chunks[0].score if chunks else 0.0
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        state["retrieved_chunks"] = []
        state["retrieval_confidence"] = 0.0
    
    return state


async def input_guard_node(state: PipelineState) -> PipelineState:
    """Input safety guardrail."""
    if not state.get("transcript"):
        state["input_guard_passed"] = False
        state["input_guard_reason"] = "empty_transcript"
        return state
    
    passed, reason = check_input_safety(state["transcript"])
    state["input_guard_passed"] = passed
    state["input_guard_reason"] = reason
    
    if not passed:
        state["refusal_reason"] = "input_guard_failed"
        lang = state.get("detected_language", "en")
        state["refusal_message"] = settings.refusal_messages.get(lang, settings.refusal_messages["en"])
    
    return state


async def retrieve_confidence_node(state: PipelineState) -> PipelineState:
    """Check retrieval confidence threshold."""
    confidence = state.get("retrieval_confidence", 0.0)
    
    if confidence < settings.retrieval_score_threshold:
        state["refusal_reason"] = "low_retrieval_confidence"
        lang = state.get("detected_language", "en")
        state["refusal_message"] = settings.refusal_messages.get(lang, settings.refusal_messages["en"])
    
    return state


async def generate_node(state: PipelineState) -> PipelineState:
    """Generate answer using Sarvam-105B."""
    if state.get("refusal_reason"):
        return state
    
    if not state.get("retrieved_chunks"):
        state["refusal_reason"] = "no_chunks"
        lang = state.get("detected_language", "en")
        state["refusal_message"] = settings.refusal_messages.get(lang, settings.refusal_messages["en"])
        return state

    try:
        # Build context
        context = "\n\n".join([
            f"Source {i+1} (query_id: {c.query_id}, selected: {c.is_selected}):\n{c.content}"
            for i, c in enumerate(state["retrieved_chunks"][:3])
        ])
        
        # Build conversation history
        history = state.get("conversation_history", [])
        history_str = "\n".join([f"{t.role}: {t.content}" for t in history[-settings.max_conversation_turns:]])
        
        # Generate - use mock if specified in state
        use_mock = state.get("use_mock_generation", False)
        print(f"DEBUG generate_node: use_mock={use_mock}, transcript={state.get('transcript')[:50]}")
        answer, reasoning = await generate_answer(
            query=state["transcript"],
            context=context,
            language=state.get("detected_language", "en"),
            history=history_str,
            use_mock=use_mock
        )
        print(f"DEBUG generate_node: answer={repr(answer[:50])}")
        state["answer"] = answer
        state["reasoning_content"] = reasoning
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        state["answer"] = ""
        state["refusal_reason"] = "generation_failed"
        lang = state.get("detected_language", "en")
        state["refusal_message"] = settings.refusal_messages.get(lang, settings.refusal_messages["en"])
    
    return state


async def grounding_node(state: PipelineState) -> PipelineState:
    """Check if answer is grounded in retrieved context."""
    if state.get("refusal_reason") or not state.get("answer"):
        state["grounded"] = False
        state["grounding_score"] = 0.0
        return state
    
    try:
        context = "\n".join([c.content for c in state["retrieved_chunks"][:3]])
        grounded, score = check_grounding(state["answer"], context)
        state["grounded"] = grounded
        state["grounding_score"] = score
        
        if not grounded:
            state["refusal_reason"] = "ungrounded"
            lang = state.get("detected_language", "en")
            state["refusal_message"] = settings.refusal_messages.get(lang, settings.refusal_messages["en"])
    except Exception as e:
        logger.error(f"Grounding check failed: {e}")
        state["grounded"] = False
        state["grounding_score"] = 0.0
    
    return state


async def hallucination_node(state: PipelineState) -> PipelineState:
    """Self-consistency hallucination check."""
    if state.get("refusal_reason") or not state.get("answer"):
        state["hallucination_check_passed"] = False
        state["hallucination_score"] = 0.0
        return state
    
    try:
        context = "\n".join([c.content for c in state["retrieved_chunks"][:3]])
        passed, score = check_hallucination(state["answer"], context)
        state["hallucination_check_passed"] = passed
        state["hallucination_score"] = score
        
        if not passed:
            state["refusal_reason"] = "hallucination_detected"
            lang = state.get("detected_language", "en")
            state["refusal_message"] = settings.refusal_messages.get(lang, settings.refusal_messages["en"])
    except Exception as e:
        logger.error(f"Hallucination check failed: {e}")
        state["hallucination_check_passed"] = False
        state["hallucination_score"] = 0.0
    
    return state


async def output_guard_node(state: PipelineState) -> PipelineState:
    """Output safety guardrail."""
    if state.get("refusal_reason"):
        # Use refusal message
        state["answer"] = state.get("refusal_message", settings.refusal_messages["en"])
        state["output_guard_passed"] = True
        return state
    
    if not state.get("answer"):
        state["output_guard_passed"] = False
        state["output_guard_reason"] = "empty_answer"
        return state
    
    passed, reason = check_output_safety(state["answer"])
    state["output_guard_passed"] = passed
    state["output_guard_reason"] = reason
    
    if not passed:
        state["refusal_reason"] = "output_guard_failed"
        lang = state.get("detected_language", "en")
        state["answer"] = settings.refusal_messages.get(lang, settings.refusal_messages["en"])
    
    return state


async def tts_node(state: PipelineState) -> PipelineState:
    """Text-to-speech using Sarvam Bulbul v3."""
    if not state.get("answer"):
        state["tts_audio"] = b""
        return state
    
    try:
        lang = state.get("detected_language", "en")
        audio = await synthesize_speech(
            state["answer"],
            language=lang,
            voice=settings.tts_voice
        )
        state["tts_audio"] = audio
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        state["tts_audio"] = b""
    
    return state


async def update_history_node(state: PipelineState) -> PipelineState:
    """Update conversation history."""
    history = state.get("conversation_history", [])
    
    if state.get("transcript"):
        history.append(ConversationTurn(
            role="user",
            content=state["transcript"],
            language=state.get("detected_language", "en")
        ))
    
    if state.get("answer") and not state.get("refusal_reason"):
        history.append(ConversationTurn(
            role="assistant",
            content=state["answer"],
            language=state.get("detected_language", "en")
        ))
    
    # Trim to max turns
    state["conversation_history"] = history[-settings.max_conversation_turns:]
    return state


# ===== CONDITIONAL ROUTING =====

def route_after_stt(state: PipelineState) -> str:
    if state.get("refusal_reason") == "stt_failed":
        return "refuse"
    return "continue"


def route_after_lang_detect(state: PipelineState) -> str:
    if state.get("refusal_reason") == "unsupported_language":
        return "refuse"
    return "continue"


def route_after_input_guard(state: PipelineState) -> str:
    if not state.get("input_guard_passed"):
        return "refuse"
    return "continue"


def route_after_retrieve_confidence(state: PipelineState) -> str:
    if state.get("refusal_reason") == "low_retrieval_confidence":
        return "refuse"
    return "continue"


def route_after_generate(state: PipelineState) -> str:
    if state.get("refusal_reason"):
        return "refuse"
    return "continue"


def route_after_grounding(state: PipelineState) -> str:
    if state.get("refusal_reason") == "ungrounded":
        return "refuse"
    return "continue"


def route_after_hallucination(state: PipelineState) -> str:
    if state.get("refusal_reason") == "hallucination_detected":
        return "refuse"
    return "continue"


def route_after_output_guard(state: PipelineState) -> str:
    return "continue"


# ===== REFUSAL NODE =====

async def refuse_node(state: PipelineState) -> PipelineState:
    """Handle refusal - generate TTS for refusal message."""
    if not state.get("answer"):
        lang = state.get("detected_language", "en")
        state["answer"] = state.get("refusal_message", settings.refusal_messages.get(lang, settings.refusal_messages["en"]))
    
    # Generate TTS for refusal
    try:
        audio = await synthesize_speech(
            state["answer"],
            language=state.get("detected_language", "en"),
            voice=settings.tts_voice
        )
        state["tts_audio"] = audio
    except Exception as e:
        logger.error(f"Refusal TTS failed: {e}")
        state["tts_audio"] = b""
    
    return state


# ===== BUILD GRAPH =====

def build_pipeline_graph() -> StateGraph:
    """Build the complete LangGraph pipeline."""
    graph = StateGraph(PipelineState)
    
    # Add nodes
    graph.add_node("stt", stt_node)
    graph.add_node("lang_detect", lang_detect_node)
    graph.add_node("embed", embed_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("retrieve_confidence", retrieve_confidence_node)
    graph.add_node("generate", generate_node)
    graph.add_node("grounding", grounding_node)
    graph.add_node("hallucination", hallucination_node)
    graph.add_node("output_guard", output_guard_node)
    graph.add_node("tts", tts_node)
    graph.add_node("update_history", update_history_node)
    graph.add_node("refuse", refuse_node)
    # Document ingestion nodes
    graph.add_node("vision_extract", vision_extract_node)
    graph.add_node("document_ingest", document_ingest_node)
    
    # Add edges
    # Document ingestion flow (when file_path provided)
    def route_after_start(state: PipelineState) -> str:
        file_path = state.get("file_path", "")
        if file_path and file_path.strip():
            return "vision_extract"
        return "stt"
    
    graph.add_conditional_edges(
        START,
        route_after_start,
        {"vision_extract": "vision_extract", "stt": "stt"}
    )
    
    # Vision extract -> document_ingest
    graph.add_edge("vision_extract", "document_ingest")
    graph.add_edge("document_ingest", END)
    
    # STT -> lang_detect or refuse
    graph.add_conditional_edges(
        "stt",
        route_after_stt,
        {"refuse": "refuse", "continue": "lang_detect"}
    )
    
    # lang_detect -> embed or refuse
    graph.add_conditional_edges(
        "lang_detect",
        route_after_lang_detect,
        {"refuse": "refuse", "continue": "embed"}
    )
    
    graph.add_edge("embed", "retrieve")
    graph.add_edge("retrieve", "input_guard")
    
    # input_guard -> retrieve_confidence or refuse
    graph.add_conditional_edges(
        "input_guard",
        route_after_input_guard,
        {"refuse": "refuse", "continue": "retrieve_confidence"}
    )
    
    # retrieve_confidence -> generate or refuse
    graph.add_conditional_edges(
        "retrieve_confidence",
        route_after_retrieve_confidence,
        {"refuse": "refuse", "continue": "generate"}
    )
    
    # generate -> grounding or refuse
    graph.add_conditional_edges(
        "generate",
        route_after_generate,
        {"refuse": "refuse", "continue": "grounding"}
    )
    
    # grounding -> hallucination or refuse
    graph.add_conditional_edges(
        "grounding",
        route_after_grounding,
        {"refuse": "refuse", "continue": "hallucination"}
    )
    
    # hallucination -> output_guard or refuse
    graph.add_conditional_edges(
        "hallucination",
        route_after_hallucination,
        {"refuse": "refuse", "continue": "output_guard"}
    )
    
    # output_guard -> tts
    graph.add_conditional_edges(
        "output_guard",
        route_after_output_guard,
        {"continue": "tts"}
    )
    
    graph.add_edge("tts", "update_history")
    graph.add_edge("update_history", END)
    graph.add_edge("refuse", END)
    
    return graph.compile()


# ===== PIPELINE RUNNER =====

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline_graph()
    return _pipeline


async def run_pipeline(
    audio_bytes: bytes,
    language: str = "en",
    session_id: str = None,
    conversation_history: List[ConversationTurn] = None,
    transcript: str = None,  # Pre-transcribed query (bypass STT)
    file_path: str = None,  # Document file for Vision OCR ingestion
    use_mock_generation: bool = False,  # Use mock generation for evaluation
) -> PipelineState:
    """Run the complete pipeline."""
    pipeline = get_pipeline()
    
    initial_state = PipelineState(
        audio_bytes=audio_bytes,
        language=language,
        session_id=session_id or str(uuid.uuid4()),
        trace_id="",
        transcript=transcript or "",
        stt_confidence=0.0,
        stt_latency_ms=0.0,
        detected_language="",
        lang_confidence=0.0,
        query_embedding=[],
        retrieved_chunks=[],
        reranked_chunks=[],
        retrieval_latency_ms=0.0,
        input_guard_passed=False,
        input_guard_reason=None,
        retrieval_confidence=0.0,
        answer="",
        generation_latency_ms=0.0,
        reasoning_content=None,
        grounded=False,
        grounding_score=0.0,
        hallucination_check_passed=False,
        hallucination_score=0.0,
        output_guard_passed=False,
        output_guard_reason=None,
        refusal_reason=None,
        refusal_message="",
        tts_audio=b"",
        tts_latency_ms=0.0,
        conversation_history=conversation_history or [],
        total_latency_ms=0.0,
        component_latencies={},
        file_path=file_path or "",
        extracted_text="",
        vision_job_id="",
        vision_output_url="",
        ingested_chunks=0,
        ingestion_language="",
        vision_error="",
        use_mock_generation=use_mock_generation,
    )
    
    print(f"DEBUG run_pipeline: Starting pipeline.ainvoke")
    start = time.perf_counter()
    result = await pipeline.ainvoke(initial_state)
    print(f"DEBUG run_pipeline: pipeline.ainvoke completed in {(time.perf_counter() - start)*1000:.1f}ms")
    result["total_latency_ms"] = (time.perf_counter() - start) * 1000
    
    print(f"DEBUG run_pipeline: Final answer: {repr(result.get('answer', '')[:50])}")
    return result