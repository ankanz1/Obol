import base64
import io
import wave
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Convert raw PCM to WAV format."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


def wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    """Convert WAV to raw PCM, return (pcm_bytes, sample_rate)."""
    buffer = io.BytesIO(wav_bytes)
    with wave.open(buffer, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.getnframes()
        pcm_data = wav_file.readframes(frames)
    
    # Convert to mono 16-bit if needed
    if channels > 1 or sample_width != 2:
        logger.warning(f"Audio format not ideal: {channels}ch, {sample_width*8}bit, {sample_rate}Hz")
    
    return pcm_data, sample_rate


def base64_to_pcm(base64_str: str) -> bytes:
    """Decode base64 to PCM bytes."""
    try:
        return base64.b64decode(base64_str)
    except Exception as e:
        logger.error(f"Base64 decode failed: {e}")
        return b""


def pcm_to_base64(pcm_bytes: bytes) -> str:
    """Encode PCM bytes to base64."""
    return base64.b64encode(pcm_bytes).decode()


def resample_audio(pcm_bytes: bytes, from_rate: int, to_rate: int = 16000) -> bytes:
    """Resample audio to target sample rate."""
    if from_rate == to_rate:
        return pcm_bytes
    
    try:
        import numpy as np
        from scipy import signal
        
        # Convert to numpy array
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Resample
        num_samples = int(len(audio) * to_rate / from_rate)
        resampled = signal.resample(audio, num_samples)
        
        # Convert back to int16
        resampled = (resampled * 32767).astype(np.int16)
        return resampled.tobytes()
        
    except Exception as e:
        logger.error(f"Resampling failed: {e}")
        return pcm_bytes


def split_audio_chunks(pcm_bytes: bytes, chunk_duration_ms: int = 100, sample_rate: int = 16000) -> list[bytes]:
    """Split PCM audio into fixed-duration chunks."""
    bytes_per_sample = 2  # 16-bit
    samples_per_chunk = int(sample_rate * chunk_duration_ms / 1000)
    bytes_per_chunk = samples_per_chunk * bytes_per_sample
    
    chunks = []
    for i in range(0, len(pcm_bytes), bytes_per_chunk):
        chunk = pcm_bytes[i:i + bytes_per_chunk]
        if len(chunk) == bytes_per_chunk:
            chunks.append(chunk)
    
    return chunks