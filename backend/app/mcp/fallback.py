"""Fallback to custom Sarvam clients when MCP fails."""
import logging
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)


async def fallback_vision_extract(file_path: str, output_format: str = "markdown") -> dict:
    """Use direct REST API for Vision OCR."""
    from app.mcp.vision_direct import vision_extract_direct
    
    fmt_map = {"markdown": "md", "html": "html", "json": "json"}
    fmt = fmt_map.get(output_format, "md")
    
    try:
        result = await vision_extract_direct(file_path, fmt)
        # Normalize to expected format
        return {
            "job_id": result.get("job_id"),
            "job_state": result.get("job_state"),
            "output_url": result.get("output_url"),
            "content": result.get("content", ""),
            "page_metrics": result.get("page_metrics"),
            "raw_status": result.get("raw_status"),
        }
    except Exception as e:
        logger.error(f"Direct Vision API failed: {e}")
        raise NotImplementedError(f"Vision extract not available: {e}")


async def fallback_identify_language(text: str) -> dict:
    """Simple language detection using existing language map."""
    # Use existing language detection heuristic
    from app.utils.language import detect_language
    lang = detect_language(text)
    return {
        "language_code": f"{lang}-IN",
        "script_code": "Devanagari" if lang in ["hi", "mr", "ne", "sa"] else "Latin"
    }


async def fallback_translate(
    text: str,
    target_language: str,
    source_language: Optional[str] = None,
    model: str = "mayura:v1",
) -> dict:
    """Translate using Sarvam chat model as fallback."""
    from app.generation.sarvam_client import generate_answer
    
    prompt = f"Translate to {target_language}: {text}"
    translated, _ = await generate_answer(prompt, "", "en")
    
    return {
        "translated_text": translated,
        "source_language": source_language or "auto",
        "target_language": target_language,
    }


async def fallback_stt_transcribe(
    audio_bytes: bytes,
    language: str = "unknown",
    mode: str = "transcribe",
) -> str:
    """Use custom STT client."""
    from app.stt.sarvam_client import transcribe_audio
    return await transcribe_audio(audio_bytes, language)


async def fallback_tts_speak(
    text: str,
    language: str = "hi-IN",
    speaker: str = "priya",
) -> bytes:
    """Use custom TTS client."""
    from app.tts.sarvam_client import synthesize_speech
    return await synthesize_speech(text, language, speaker)


async def fallback_llm_complete(
    messages: list[dict],
    model: str = "sarvam-105b",
    temperature: float = 0.2,
) -> str:
    """Use custom chat client."""
    from app.generation.sarvam_client import generate_answer
    
    # Convert messages to query + context
    query = messages[-1].get("content", "") if messages else ""
    context = "\n".join([m.get("content", "") for m in messages[:-1]])
    
    answer, _ = await generate_answer(query, context, "en")
    return answer


class MCPFallback:
    """Wrapper that tries MCP first, falls back to custom clients."""
    
    @staticmethod
    async def vision_extract(file_path: str, output_format: str = "markdown") -> dict:
        # Skip MCP - use direct REST API directly
        return await fallback_vision_extract(file_path, output_format)
    
    @staticmethod
    async def identify_language(text: str) -> dict:
        try:
            from app.mcp.tools import mcp_identify_language
            return await mcp_identify_language(text)
        except Exception as e:
            logger.warning(f"MCP identify_language failed, using fallback: {e}")
            return await fallback_identify_language(text)
    
    @staticmethod
    async def translate(
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
        model: str = "mayura:v1",
    ) -> dict:
        try:
            from app.mcp.tools import mcp_translate
            return await mcp_translate(text, target_language, source_language, model)
        except Exception as e:
            logger.warning(f"MCP translate failed, using fallback: {e}")
            return await fallback_translate(text, target_language, source_language, model)
    
    @staticmethod
    async def stt_transcribe(
        audio_bytes: bytes,
        language: str = "unknown",
        mode: str = "transcribe",
    ) -> str:
        try:
            from app.mcp.tools import mcp_stt_transcribe
            return await mcp_stt_transcribe(audio_bytes, language, mode)
        except Exception as e:
            logger.warning(f"MCP STT failed, using custom client: {e}")
            return await fallback_stt_transcribe(audio_bytes, language, mode)
    
    @staticmethod
    async def tts_speak(
        text: str,
        language: str = "hi-IN",
        speaker: str = "priya",
    ) -> bytes:
        try:
            from app.mcp.tools import mcp_tts_speak
            return await mcp_tts_speak(text, language, speaker)
        except Exception as e:
            logger.warning(f"MCP TTS failed, using custom client: {e}")
            return await fallback_tts_speak(text, language, speaker)
    
    @staticmethod
    async def llm_complete(
        messages: list[dict],
        model: str = "sarvam-105b",
        temperature: float = 0.2,
    ) -> str:
        try:
            from app.mcp.tools import mcp_llm_complete
            return await mcp_llm_complete(messages, model, temperature)
        except Exception as e:
            logger.warning(f"MCP LLM failed, using custom client: {e}")
            return await fallback_llm_complete(messages, model, temperature)