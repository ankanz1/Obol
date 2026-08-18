"""MCP adapter for Sarvam AI tools."""
from .client import SarvamMCPClient, get_mcp_client
from .tools import (
    mcp_vision_extract,
    mcp_identify_language,
    mcp_translate,
    mcp_stt_transcribe,
    mcp_tts_speak,
    mcp_llm_complete,
)
from .fallback import MCPFallback

__all__ = [
    "SarvamMCPClient",
    "get_mcp_client",
    "mcp_vision_extract",
    "mcp_identify_language",
    "mcp_translate",
    "mcp_stt_transcribe",
    "mcp_tts_speak",
    "mcp_llm_complete",
    "MCPFallback",
]