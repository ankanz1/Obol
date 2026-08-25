# Voice RAG for MSMARCO-XI (18 Indic Languages)

A voice-enabled Retrieval-Augmented Generation system supporting 18 Indian languages + English. Built with Sarvam AI (STT, LLM, TTS), Qdrant vector database, and LangGraph orchestration.

## 🎯 Features

- **Speech-to-Text**: Sarvam Saaras v3 (12+ Indic languages)
- **LLM**: Sarvam-105B (128K context, Indic-optimized)
- **Text-to-Speech**: Sarvam Bulbul v3 (30+ voices)
- **Vector Search**: Qdrant with HNSW (18 language-specific collections)
- **Embeddings**: multilingual-e5-small (384-dim, ONNX for speed)
- **Guardrails**: Input safety, grounding check (NLI), hallucination detection
- **Latency Target**: <200ms P50 end-to-end
- **Frontend**: React + WebRTC voice UI with waveform visualization

## 🗣️ Supported Languages (18 Indic + English)

| Code | Language | Native | Code | Language | Native |
|------|----------|--------|------|----------|--------|
| hi | Hindi | हिंदी | bn | Bengali | বাংলা |
| ta | Tamil | தமிழ் | te | Telugu | తెలుగు |
| mr | Marathi | मराठी | gu | Gujarati | ગુજરાતી |
| kn | Kannada | ಕನ್ನಡ | ml | Malayalam | മലയാളം |
| pa | Punjabi | ਪੰਜਾਬੀ | or | Odia | ଓଡ଼ିଆ |
| as | Assamese | অসমীয়া | ur | Urdu | اردو |
| ne | Nepali | नेपाली | sa | Sanskrit | संस्कृतम् |
| ks | Kashmiri | کٲشُر | sd | Sindhi | سنڌي |
| doi | Dogri | डोगरी | sat | Santali | ᱥᱟᱱᱛᱟᱲᱤ |
| en | English | English | | | |

## 🏗️ Architecture

```
Voice Input → WebRTC → Sarvam STT → Language Detection → Embedding (ONNX)
                                                      ↓
                                              Qdrant Search (per-language)
                                                      ↓
                                              Guardrails (Input)
                                                      ↓
                                              Sarvam-105B Generation
                                                      ↓
                                              Grounding Check (NLI)
                                                      ↓
                                              Hallucination Check
                                                      ↓
                                              Guardrails (Output)
                                                      ↓
                                              Sarvam TTS (Bulbul v3)
                                                      ↓
                                              Audio Playback
```

## 📦 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Sarvam API key (from [dashboard.sarvam.ai](https://dashboard.sarvam.ai))
- Qdrant Cloud instance (or local Docker)

### Installation

```bash
# Clone and setup
git clone <your-repo>
cd voice-rag-msmarco

# Run setup script
./setup.sh

# Edit .env with your API keys
nano .env
```

### Build Index (First Time - Takes 1-2 Hours)

```bash
# Full pipeline: download → chunk → embed → index
cd backend
python scripts/build_index.py --full --recreate
```

Or run steps separately:
```bash
# 1. Download dataset (streaming, low memory)
python data/download.py

# 2. Chunk with 5 strategies
python -m data.chunking.pipeline

# 3. Build Qdrant index
python scripts/build_index.py --recreate
```

### Run Services

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
python -m app.main
# API at http://localhost:8000
# WebSocket at ws://localhost:8000/ws/{session_id}
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
# UI at http://localhost:3000
```

## Configuration

Edit `.env`:
```bash
# Required
SARVAM_API_KEY=sk_xxx
QDRANT_URL=https://xxx.cloud.qdrant.io
QDRANT_API_KEY=xxx

# Optional overrides
EMBEDDING_MODEL=intfloat/multilingual-e5-small
CHAT_MODEL=sarvam-105b
TTS_MODEL=bulbul:v3
TTS_VOICE=anushka
RETRIEVAL_LIMIT=5
RETRIEVAL_SCORE_THRESHOLD=0.65
GROUNDING_THRESHOLD=0.7
```

## Latency Evaluation

```bash
cd backend
python evaluation/latency_harness.py
# Results in evaluation/results.json
```

Outputs P50/P70/P90/P99/P100 latencies by language, query type, and component.

## Project Structure

```
voice-rag-msmarco/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI + WebSocket
│   │   ├── config.py            # Settings
│   │   ├── pipeline/
│   │   │   ├── graph.py         # LangGraph orchestration
│   │   │   └── schemas.py       # Pydantic models
│   │   ├── retrieval/
│   │   │   ├── embedder.py      # ONNX embedder
│   │   │   └── qdrant_client.py # Vector search
│   │   ├── generation/
│   │   │   └── sarvam_client.py # Sarvam-105B wrapper
│   │   ├── stt/tts/
│   │   │   └── sarvam_client.py # Sarvam STT/TTS
│   │   ├── guardrails/          # Safety, grounding, hallucination
│   │   └── utils/               # Language, audio
│   ├── data/
│   │   ├── download.py          # HF dataset loader
│   │   ├── chunking/            # 5 chunking strategies
│   │   └── index_builder.py     # Qdrant upsert
│   ├── evaluation/
│   │   └── latency_harness.py   # P50/P70/P100 measurement
│   └── scripts/
│       └── build_index.py       # One-command index build
├── frontend/
│   ├── src/
│   │   ├── components/          # VoiceRecorder, Waveform, etc.
│   │   ├── hooks/               # useWebSocket, useAudioRecorder
│   │   └── App.tsx              # Main UI
│   └── package.json
├── .env.example
├── setup.sh
└── README.md
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + collection stats |
| POST | `/api/query` | REST voice query (base64 audio) |
| WS | `/ws/{session_id}` | Real-time voice interaction |

### WebSocket Protocol

```json
// Send audio
{"type": "audio", "audio_base64": "...", "language": "hi"}

// Config
{"type": "config", "language": "ta"}

// Receive result
{"type": "result", "transcript": "...", "answer": "...", 
 "audio_base64": "...", "language": "hi", "grounded": true,
 "latency_ms": 187, "trace_id": "abc123"}
```

## Chunking Strategies (5)

1. **Passage-Level**: Each passage as chunk (~11M chunks)
2. **Query-Passage Pair**: Query + selected passage (~1M)
3. **Sliding Window**: 512-token windows, 100 overlap (~500K)
4. **Language-Routed**: Per-language Qdrant collections
5. **Metadata-Enriched**: Rich metadata for filtering/reranking

All strategies applied → unified index with `source` field.

##  Guardrails

| Layer | Method | Threshold |
|-------|--------|-----------|
| Input Safety | Regex + keyword | PII, injection, toxicity |
| Language Gate | FastText lid.176 | Top-1 prob > 0.5 |
| Retrieval Confidence | Cosine similarity | > 0.65 |
| Grounding (NLI) | DeBERTa-v3-small MNLI | Entailment > 0.7 |
| Hallucination | Self-consistency (n=2) | Jaccard > 0.6 |
| Output Safety | Regex | PII, toxicity |

## Latency Optimization

- **ONNX Runtime** for embeddings/NLI (CPU/GPU)
- **Connection pooling** for Qdrant/Sarvam
- **Streaming** STT/TTS via WebSocket
- **Parallel guardrails** (input + retrieval confidence)
- **Speculative embedding** during STT
- **Caching** frequent queries (optional)

## GPU 

# Install GPU deps
pip install onnxruntime-gpu torch --index-url https://download.pytorch.org/whl/cu121

# Run evaluation (fast on GPU)
python backend/evaluation/latency_harness.py
```
