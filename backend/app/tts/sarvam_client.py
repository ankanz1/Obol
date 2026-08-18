import asyncio
import base64
import json
import logging
import websockets
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


class SarvamTTSClient:
    """Sarvam Bulbul v3 Streaming TTS via WebSocket."""
    
    def __init__(self):
        self.api_key = settings.sarvam_api_key
        self.model = settings.tts_model
        self.voice = settings.tts_voice
    
    async def synthesize_streaming(
        self,
        text: str,
        language: str = "en",
        voice: str = None
    ) -> bytes:
        """Stream text and receive audio chunks."""
        voice = voice or self.voice
        uri = f"wss://api.sarvam.ai/v1/text-to-speech/streaming?model={self.model}"
        headers = {"api-subscription-key": self.api_key}
        
        audio_chunks = []
        
        async with websockets.connect(uri, extra_headers=headers) as ws:
            # Send text
            msg = {
                "text": text,
                "target_language_code": self._lang_to_code(language),
                "speaker": voice,
                "sample_rate": 16000,
                "enable_preprocessing": True
            }
            await ws.send(json.dumps(msg))
            
            # Receive audio chunks
            try:
                async for msg in ws:
                    data = json.loads(msg)
                    
                    if data.get("event") == "audio_chunk":
                        chunk_b64 = data.get("audio_base_64", "")
                        if chunk_b64:
                            audio_chunks.append(base64.b64decode(chunk_b64))
                    elif data.get("event") == "end":
                        break
                    elif data.get("event") == "error":
                        logger.error(f"TTS error: {data.get('error')}")
                        break
            except Exception as e:
                logger.error(f"TTS streaming error: {e}")
        
        return b"".join(audio_chunks)
    
    def _lang_to_code(self, lang: str) -> str:
        mapping = {
            "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN",
            "mr": "mr-IN", "gu": "gu-IN", "kn": "kn-IN", "ml": "ml-IN",
            "pa": "pa-IN", "or": "od-IN", "as": "as-IN", "ur": "ur-IN",
            "ne": "ne-IN", "sa": "sa-IN", "ks": "ks-IN", "sd": "sd-IN",
            "doi": "doi-IN", "sat": "sat-IN", "en": "en-IN"
        }
        return mapping.get(lang, "en-IN")


class SarvamTTSRestClient:
    """Sarvam Bulbul v3 REST API."""
    
    def __init__(self):
        from sarvamai import SarvamAI
        self.client = SarvamAI(api_subscription_key=settings.sarvam_api_key)
        self.voice = settings.tts_voice
    
    async def synthesize(
        self,
        text: str,
        language: str = "en",
        voice: str = None
    ) -> bytes:
        """Synthesize speech via REST API."""
        try:
            voice = voice or self.voice
            
            response = self.client.text_to_speech.convert(
                model=settings.tts_model,
                text=text,
                language_code=self._lang_to_code(language),
                speaker=voice,
            )
            
            # Response contains list of base64 audio strings
            import base64
            if response.audios and len(response.audios) > 0:
                return base64.b64decode(response.audios[0])
            return b""
            
        except Exception as e:
            logger.warning(f"REST TTS failed (API may not be available): {e}")
            # Return empty for mock/testing
            return b""
    
    def _lang_to_code(self, lang: str) -> str:
        mapping = {
            "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN",
            "mr": "mr-IN", "gu": "gu-IN", "kn": "kn-IN", "ml": "ml-IN",
            "pa": "pa-IN", "or": "od-IN", "as": "as-IN", "ur": "ur-IN",
            "ne": "ne-IN", "sa": "sa-IN", "ks": "ks-IN", "sd": "sd-IN",
            "doi": "doi-IN", "sat": "sat-IN", "en": "en-IN"
        }
        return mapping.get(lang, "en-IN")


# Use REST for simplicity
_tts_client = None

def get_tts_client():
    global _tts_client
    if _tts_client is None:
        _tts_client = SarvamTTSRestClient()
    return _tts_client


async def synthesize_speech(
    text: str,
    language: str = "en",
    voice: str = None
) -> bytes:
    """Synthesize speech using Sarvam TTS."""
    client = get_tts_client()
    return await client.synthesize(text, language, voice)