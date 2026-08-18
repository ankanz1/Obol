import asyncio
import base64
import json
import logging
import websockets
from typing import Optional, Tuple
import uuid
import io
import wave

from app.config import settings

logger = logging.getLogger(__name__)


class SarvamSTTClient:
    """Sarvam Saaras v3 Realtime STT via WebSocket."""
    
    def __init__(self):
        self.api_key = settings.sarvam_api_key
        self.model = settings.stt_model
        self.sample_rate = settings.sample_rate
    
    async def transcribe_streaming(
        self,
        audio_chunks: asyncio.Queue,
        language: str = "en"
    ) -> Tuple[str, float]:
        """Stream audio chunks and get transcript."""
        uri = f"wss://api.sarvam.ai/v1/speech-to-text/realtime?model={self.model}"
        headers = {"api-subscription-key": self.api_key}
        
        transcript = ""
        confidence = 0.0
        final_received = asyncio.Event()
        
        async with websockets.connect(uri, extra_headers=headers) as ws:
            # Send session config
            config = {
                "message_type": "session_started",
                "config": {
                    "sample_rate": self.sample_rate,
                    "audio_format": "pcm_16000",
                    "language_code": self._lang_to_code(language),
                    "commit_strategy": "vad",
                    "vad_threshold": 0.5,
                    "vad_silence_threshold_secs": 0.8,
                    "min_speech_duration_ms": 100,
                    "min_silence_duration_ms": 500,
                    "enable_logging": False
                }
            }
            await ws.send(json.dumps(config))
            
            # Receive session started
            await ws.recv()
            
            async def send_audio():
                while True:
                    chunk = await audio_chunks.get()
                    if chunk is None:  # End signal
                        break
                    
                    msg = {
                        "message_type": "input_audio_chunk",
                        "audio_base_64": base64.b64encode(chunk).decode(),
                        "commit": False,
                        "sample_rate": self.sample_rate
                    }
                    await ws.send(json.dumps(msg))
                
                # Send final commit
                await ws.send(json.dumps({
                    "message_type": "input_audio_chunk",
                    "audio_base_64": "",
                    "commit": True,
                    "sample_rate": self.sample_rate
                }))
            
            async def receive_transcript():
                nonlocal transcript, confidence
                try:
                    async for msg in ws:
                        data = json.loads(msg)
                        msg_type = data.get("message_type")
                        
                        if msg_type == "partial_transcript":
                            transcript = data.get("text", "")
                        elif msg_type in ("committed_transcript", "final_transcript"):
                            transcript = data.get("text", "")
                            confidence = 1.0  # Sarvam doesn't return confidence in realtime
                            final_received.set()
                            break
                        elif msg_type == "error":
                            logger.error(f"STT error: {data.get('error')}")
                            break
                except Exception as e:
                    logger.error(f"STT receive error: {e}")
            
            await asyncio.gather(send_audio(), receive_transcript())
        
        return transcript, confidence
    
    def _lang_to_code(self, lang: str) -> str:
        """Map our lang codes to Sarvam language codes."""
        mapping = {
            "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN",
            "mr": "mr-IN", "gu": "gu-IN", "kn": "kn-IN", "ml": "ml-IN",
            "pa": "pa-IN", "or": "od-IN", "as": "as-IN", "ur": "ur-IN",
            "ne": "ne-IN", "sa": "sa-IN", "ks": "ks-IN", "sd": "sd-IN",
            "doi": "doi-IN", "sat": "sat-IN", "en": "en-IN"
        }
        return mapping.get(lang, "en-IN")


class SarvamSTTRestClient:
    """Sarvam Saaras v3 REST API for shorter audio (<30s)."""
    
    def __init__(self):
        from sarvamai import SarvamAI
        self.client = SarvamAI(api_subscription_key=settings.sarvam_api_key)
    
    def _create_wav(self, pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
        """Create WAV file from PCM bytes."""
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        return buffer.getvalue()
    
    async def transcribe(self, audio_bytes: bytes, language: str = "en") -> Tuple[str, float]:
        """Transcribe audio file via REST API."""
        try:
            # Create WAV file from PCM bytes
            wav_bytes = self._create_wav(audio_bytes)
            file_obj = io.BytesIO(wav_bytes)
            
            response = self.client.speech_to_text.transcribe(
                file=file_obj,
                model=settings.stt_model.replace("-realtime", ""),
                mode="transcribe",
                language_code=self._lang_to_code(language)
            )
            
            return response.transcript, 0.9  # Approximate confidence
            
        except Exception as e:
            logger.warning(f"REST STT failed (API may not be available): {e}")
            # Return empty transcript for mock/testing
            return "", 0.0
    
    def _lang_to_code(self, lang: str) -> str:
        mapping = {
            "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN",
            "mr": "mr-IN", "gu": "gu-IN", "kn": "kn-IN", "ml": "ml-IN",
            "pa": "pa-IN", "or": "od-IN", "as": "as-IN", "ur": "ur-IN",
            "ne": "ne-IN", "sa": "sa-IN", "ks": "ks-IN", "sd": "sd-IN",
            "doi": "doi-IN", "sat": "sat-IN", "en": "en-IN"
        }
        return mapping.get(lang, "en-IN")


# Use REST for simplicity (better for short queries)
_stt_client = None

def get_stt_client():
    global _stt_client
    if _stt_client is None:
        _stt_client = SarvamSTTRestClient()
    return _stt_client


async def transcribe_audio(audio_bytes: bytes, language: str = "en") -> Tuple[str, float]:
    """Transcribe audio using Sarvam STT."""
    client = get_stt_client()
    return await client.transcribe(audio_bytes, language)