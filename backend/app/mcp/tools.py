"""High-level wrappers for Sarvam MCP tools."""
import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from app.config import settings
from app.mcp.client import SarvamMCPClient, get_mcp_client

logger = logging.getLogger(__name__)


async def _call_tool(name: str, arguments: dict) -> Any:
    """Call MCP tool with error handling."""
    client = get_mcp_client()
    try:
        return await client.call_tool(name, arguments)
    except Exception as e:
        logger.error(f"MCP tool {name} failed: {e}")
        raise


async def mcp_vision_extract(
    file_path: str,
    output_format: str = "markdown",
) -> dict:
    """
    Extract text + structure from document/image using Sarvam Vision.
    
    Args:
        file_path: Path to PDF/image file
        output_format: "markdown" | "html" | "json"
    
    Returns:
        Dict with 'output_url', 'job_id', 'pages', etc.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    result = await _call_tool("sarvam_tools_vision_extract", {
        "file_path": str(path.absolute()),
        "output_format": output_format,
    })
    
    # Parse result - it's a JSON string
    return json.loads(result) if isinstance(result, str) else result


async def mcp_identify_language(text: str) -> dict:
    """
    Detect language and script of input text.
    
    Returns:
        Dict with 'language_code' (e.g. 'hi-IN'), 'script_code' (e.g. 'Devanagari')
    """
    result = await _call_tool("sarvam_tools_identify_language", {
        "text": text,
    })
    return json.loads(result) if isinstance(result, str) else result


async def mcp_translate(
    text: str,
    target_language: str,
    source_language: Optional[str] = None,
    model: str = "mayura:v1",
) -> dict:
    """
    Translate text between English and Indian languages.
    
    Args:
        text: Text to translate
        target_language: BCP-47 code (e.g. 'hi-IN', 'en-IN')
        source_language: Auto-detected if not provided
        model: 'mayura:v1' (11 langs, stylized) or 'sarvam-translate:v1' (22 langs, formal)
    
    Returns:
        Dict with 'translated_text', 'source_language', 'target_language'
    """
    args = {
        "text": text,
        "target_language_code": target_language,
        "model": model,
    }
    if source_language:
        args["source_language_code"] = source_language
    
    result = await _call_tool("sarvam_tools_translate", args)
    return json.loads(result) if isinstance(result, str) else result


async def mcp_stt_transcribe(
    audio_bytes: bytes,
    language: str = "unknown",
    mode: str = "transcribe",
) -> str:
    """
    Transcribe audio file using Sarvam STT.
    
    Args:
        audio_bytes: Raw audio data (WAV/MP3)
        language: BCP-47 code (e.g. 'hi-IN') or 'unknown' for auto-detect
        mode: 'transcribe' | 'translate' | 'verbatim' | 'translit' | 'codemix'
    
    Returns:
        Transcript text
    """
    # Save audio to temp file (MCP expects file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name
    
    try:
        result = await _call_tool("sarvam_tools_stt_transcribe", {
            "file_path": temp_path,
            "language_code": language,
            "mode": mode,
        })
        return result
    finally:
        os.unlink(temp_path)


async def mcp_tts_speak(
    text: str,
    language: str = "hi-IN",
    speaker: str = "priya",
) -> bytes:
    """
    Generate speech from text using Sarvam TTS.
    
    Args:
        text: Text to synthesize
        language: BCP-47 code (e.g. 'hi-IN')
        speaker: Voice name (priya, aditya, rahul, etc.)
    
    Returns:
        Audio bytes (WAV)
    """
    # MCP saves to file, we read it back
    result = await _call_tool("sarvam_tools_tts_speak", {
        "text": text,
        "target_language_code": language,
        "speaker": speaker,
    })
    
    # Result is file path or resource URI
    file_path = result
    if isinstance(result, str) and result.startswith("sarvam://"):
        # Extract path from URI
        file_path = result.replace("sarvam://", "")
    
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    
    os.unlink(file_path)
    return audio_bytes


async def mcp_llm_complete(
    messages: list[dict],
    model: str = "sarvam-105b",
    temperature: float = 0.2,
) -> str:
    """
    Generate chat completion with Sarvam LLM.
    
    Args:
        messages: OpenAI format [{'role': 'user', 'content': '...'}]
        model: 'sarvam-105b' (flagship) or 'sarvam-30b' (lighter)
        temperature: Sampling temperature
    
    Returns:
        Assistant response text
    """
    result = await _call_tool("sarvam_tools_llm_complete", {
        "messages": messages,
        "model": model,
        "temperature": temperature,
    })
    return result