#!/usr/bin/env python3
"""Quick test to verify imports and configuration."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Test config
from app.config import settings
print(f"✅ Config loaded")
print(f"   Sarvam key: {'set' if settings.sarvam_api_key else 'NOT SET'}")
print(f"   Qdrant URL: {settings.qdrant_url or 'NOT SET'}")
print(f"   Supported languages: {len(settings.supported_languages)}")

# Test schemas
from app.pipeline.schemas import PipelineState, PipelineInput, PipelineOutput
print(f"✅ Schemas loaded")

# Test chunking
from data.chunking.strategies import get_all_chunkers, Chunk
chunkers = get_all_chunkers()
print(f"✅ Chunking strategies: {len(chunkers)} ({[c.__class__.__name__ for c in chunkers]})")

# Test embedder (lazy load)
try:
    from app.retrieval.embedder import get_embedder
    embedder = get_embedder()
    print(f"✅ Embedder loaded: {embedder.model_path}")
except Exception as e:
    print(f"⚠️  Embedder not loaded (needs model download): {e}")

# Test Qdrant client
try:
    from app.retrieval.qdrant_client import get_qdrant
    qdrant = get_qdrant()
    print(f"✅ Qdrant client initialized")
except Exception as e:
    print(f"⚠️  Qdrant client issue: {e}")

# Test guardrails
from app.guardrails.input_guard import check_input_safety
result = check_input_safety("Hello world")
print(f"✅ Input guard: {result}")

from app.guardrails.output_guard import check_output_safety
result = check_output_safety("Hello world")
print(f"✅ Output guard: {result}")

# Test language detection
try:
    from app.utils.language import detect_language
    lang, conf = detect_language("Hello world")
    print(f"✅ Language detection: {lang} ({conf:.2f})")
except Exception as e:
    print(f"⚠️  Language detection: {e}")

print("\n✅ All basic imports working!")