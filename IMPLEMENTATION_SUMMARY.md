# Voice RAG for MSMARCO-XI - Implementation Summary

## ✅ Completed Components

### 1. **Project Structure**
```
voice_rag_msmarco/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI + WebSocket server
│   │   ├── config.py            # Pydantic settings (API keys, models, thresholds)
│   │   ├── pipeline/
│   │   │   ├── graph.py         # LangGraph orchestration (12 nodes)
│   │   │   └── schemas.py       # Pydantic I/O models
│   │   ├── retrieval/
│   │   │   ├── embedder.py      # Transformers-based embedder (multilingual-e5-small)
│   │   │   └── qdrant_client.py # Qdrant manager (18 language collections)
│   │   ├── generation/
│   │   │   └── sarvam_client.py # Sarvam-105B wrapper (streaming, structured)
│   │   ├── stt/
│   │   │   └── sarvam_client.py # Sarvam Saaras v3 (REST + WebSocket)
│   │   ├── tts/
│   │   │   └── sarvam_client.py # Sarvam Bulbul v3 (REST + WebSocket)
│   │   ├── guardrails/
│   │   │   ├── input_guard.py   # PII, injection, toxicity detection
│   │   │   ├── grounding.py     # NLI entailment check (DeBERTa-v3-small)
│   │   │   ├── hallucination.py # Self-consistency check
│   │   │   └── output_guard.py  # Output safety validation
│   │   └── utils/
│   │       ├── language.py      # Language detection (FastText + fallback)
│   │       └── audio.py         # PCM/WAV conversion utilities
│   ├── data/
│   │   ├── download.py          # HF dataset streaming downloader
│   │   ├── chunking/
│   │   │   ├── strategies.py    # 5 chunking strategies
│   │   │   └── pipeline.py      # Batch processing pipeline
│   │   └── index_builder.py     # Qdrant index builder
│   ├── evaluation/
│   │   └── latency_harness.py   # P50/P70/P90/P99/P100 measurement
│   └── scripts/
│       ├── build_index.py       # One-command index build
│       └── download_model.py    # Model downloader
├── frontend/
│   ├── src/
│   │   ├── components/          # VoiceRecorder, Waveform, MessageList, etc.
│   │   ├── hooks/               # useWebSocket, useAudioRecorder, useAudioPlayer
│   │   └── App.tsx              # Main React voice UI
│   └── package.json
├── docker-compose.yml           # Local Qdrant
├── .env.example                 # Config template
└── README.md                    # Documentation
```

### 2. **Pipeline Orchestration (LangGraph)**
12-node graph with conditional routing:
- `stt` → `lang_detect` → `embed` → `retrieve` → `input_guard` → `retrieve_confidence`
- → `generate` → `grounding` → `hallucination` → `output_guard` → `tts` → `update_history`
- Refusal paths at each guardrail layer

### 3. **Chunking Strategies (5)**
| Strategy | Description |
|----------|-------------|
| Passage-Level | Each passage as separate chunk (~11M) |
| Query-Passage Pair | Query + selected passage (~1M) |
| Sliding Window | 512-token windows, 100 overlap (~500K) |
| Language-Routed | Per-language Qdrant collections |
| Metadata-Enriched | Rich metadata for filtering/reranking |

### 4. **Guardrails (6 Layers)**
1. **Language Gate** - FastText detection, rejects unsupported
2. **Input Safety** - PII, injection, toxicity patterns
3. **Retrieval Confidence** - Cosine > 0.65 threshold
4. **Grounding (NLI)** - DeBERTa-v3-small entailment > 0.7
5. **Hallucination** - Self-consistency (Jaccard > 0.6)
6. **Output Safety** - PII, toxicity in generated response

### 5. **Sarvam AI Integration**
| Component | Model | API |
|-----------|-------|-----|
| STT | Saaras v3 | REST + WebSocket |
| LLM | Sarvam-105B | Chat completions (streaming, tools) |
| TTS | Bulbul v3 | REST + WebSocket (voice: aditya) |

### 6. **React Voice UI**
- WebRTC audio capture (16kHz mono)
- Real-time waveform visualization
- Language selector (18 Indic + English)
- Streaming audio playback
- Conversation history with grounding badges

### 7. **Latency Evaluation**
- P50/P70/P90/P95/P99/P100 percentiles
- Component-level breakdown (STT, embed, retrieve, generate, grounding, TTS)
- Per-language and per-query-type breakdown
- Bootstrap 95% CI

## ⚙️ Configuration (.env)
```bash
# Required
SARVAM_API_KEY=sk_xxx
QDRANT_URL=http://localhost:6333  # Local dev
QDRANT_API_KEY=                   # Optional for local

# Optional overrides
# EMBEDDING_MODEL=intfloat/multilingual-e5-small
# CHAT_MODEL=sarvam-105b
# TTS_MODEL=bulbul:v3
# TTS_VOICE=aditya
```

## 🚀 Running the System

### Prerequisites
```bash
# 1. Start local Qdrant
docker-compose up -d qdrant

# 2. Install dependencies
cd backend && pip install -r requirements.txt
cd frontend && npm install

# 3. Build index (first time - downloads 11M rows)
cd backend && PYTHONPATH=. python scripts/build_index.py --recreate

# 4. Start backend
cd backend && PYTHONPATH=. python -m app.main

# 5. Start frontend
cd frontend && npm run dev
```

### API Endpoints
- `GET /health` - Health check + collection stats
- `POST /api/query` - REST voice query (base64 audio)
- `WS /ws/{session_id}` - Real-time voice interaction

## 📊 Test Results (Components)
- ✅ Embedding: multilingual-e5-small (384-dim) loaded
- ✅ Generation: Sarvam-105B streaming working
- ✅ TTS: Bulbul v3 (voice: aditya) → 35KB audio
- ✅ STT: Saaras v3 REST working
- ✅ Guardrails: Input/output safety, NLI grounding, hallucination check
- ✅ Language detection: FastText fallback (script-based) working
- ✅ React UI: Components, hooks, WebSocket integration complete

## ⚠️ Known Issues / TODO
1. **Qdrant Connection** - User's cloud instance (403 Forbidden). Use local Qdrant via docker-compose.
2. **Sarvam API Key** - User provided key may have rate limits. Monitor usage.
3. **Full Index Build** - Downloads 11M rows from HF. Takes 1-2 hours on CPU.
4. **GPU Acceleration** - Embedding/NLI can use GPU (onnxruntime-gpu, CUDA).
5. **FastText** - Optional install for better language detection.

## 🎯 Next Steps
1. Run `docker-compose up -d qdrant` for local vector DB
2. Execute `PYTHONPATH=. python backend/scripts/build_index.py --recreate` to build index
3. Start backend: `cd backend && PYTHONPATH=. python -m app.main`
4. Start frontend: `cd frontend && npm run dev`
5. Open http://localhost:3000 and test voice queries in 18 languages!

## 📁 Key Files for Customization
- `backend/app/config.py` - All thresholds, model names, language list
- `backend/app/pipeline/graph.py` - Pipeline logic, routing
- `backend/app/guardrails/` - Safety thresholds
- `frontend/src/components/` - UI components
- `backend/data/chunking/strategies.py` - Chunking logic