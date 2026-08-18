"""Async MCP client for Sarvam AI server."""
import asyncio
import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()


class SarvamMCPClient:
    """Async wrapper around Sarvam MCP server via stdio."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_path: Optional[str] = None,
        output_mode: str = "files",
    ):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        self.base_path = base_path or os.getenv("SARVAM_MCP_BASE_PATH", "/tmp/sarvam_mcp")
        self.output_mode = output_mode
        self._session: Optional[ClientSession] = None
        self._stdio_cm = None
        self._read = None
        self._write = None
        self._connected = False

    async def connect(self) -> None:
        """Start MCP server and initialize session."""
        if self._connected:
            return

        env = os.environ.copy()
        env["SARVAM_API_KEY"] = self.api_key or ""
        env["SARVAM_MCP_BASE_PATH"] = self.base_path
        env["SARVAM_AUDIO_OUTPUT_MODE"] = self.output_mode

        server_params = StdioServerParameters(
            command="python3",
            args=["-m", "sarvam_mcp"],
            env=env,
        )

        self._stdio_cm = stdio_client(server_params)
        self._read, self._write = await self._stdio_cm.__aenter__()
        self._session = ClientSession(self._read, self._write)
        await self._session.initialize()
        self._connected = True
        logger.info("Sarvam MCP client connected")

    async def close(self) -> None:
        """Close MCP connection."""
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass
        if self._stdio_cm:
            try:
                await self._stdio_cm.__aexit__(None, None, None)
            except Exception:
                pass
        self._connected = False

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """Call an MCP tool by name."""
        if not self._connected:
            await self.connect()

        result = await self._session.call_tool(name, arguments)
        return result.content[0].text if result.content else None


_mcp_client: Optional[SarvamMCPClient] = None


def get_mcp_client() -> SarvamMCPClient:
    """Get or create global MCP client."""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = SarvamMCPClient()
    return _mcp_client