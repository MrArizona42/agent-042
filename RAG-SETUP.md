# RAG System Setup Instructions

This document describes how to set up and use the baseline RAG (Retrieval Augmented Generation) system for agent-042.

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [What's Implemented](#whats-implemented)
4. [What's Postponed](#whats-postponed)
5. [Setup Instructions](#setup-instructions)
6. [Running the System](#running-the-system)
7. [Testing RAG](#testing-rag)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The baseline RAG system provides context-aware responses by retrieving relevant information from a knowledge base before generating answers. The system is optimized to run on RTX 3060 (12GB VRAM) alongside the base LLM.

**Key Design Decisions:**
- **Vector DB**: Qdrant (production-ready, better than ChromaDB for scaling)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (~80MB, runs on CPU)
- **Collections**: Separate collections for `chat` (ArXiv papers) and `code` (PyTorch docs)
- **CPU Embeddings**: Embedding model runs on CPU to save GPU VRAM for vLLM

---

## System Requirements

### Hardware
- **GPU**: RTX 3060 12GB VRAM (or equivalent)
- **RAM**: 16GB+ recommended
- **Storage**: ~5GB for data + indices

### Software
- Docker & Docker Compose
- Python 3.12+
- Git

### VRAM Allocation (RTX 3060 12GB)
```
Total: 12GB
├─ vLLM + Base Model: ~8-9GB (with quantization)
├─ Embedding Model (CPU): 0GB
├─ OS/Other: ~2-3GB
└─ Available: ~1GB buffer
```

---

## What's Implemented

### ✅ Core RAG Infrastructure

**1. Vector Database (Qdrant)**
- Dockerized Qdrant service
- Persistent storage
- REST API on port 6333

**2. Embedding Service**
- `sentence-transformers/all-MiniLM-L6-v2` model
- CPU-based to preserve GPU for vLLM
- 384-dimensional vectors
- Fast inference (~50ms per query)

**3. Document Chunking**
- **Fixed Token Chunking**: Baseline for all documents (512 chars, 50 overlap)
- **Code Chunking**: Preserves functions/classes for code docs
- **Section-Aware Chunking**: Respects markdown structure for papers

**4. Retrieval Service**
- Dense retrieval using cosine similarity
- Top-k retrieval (default: 5 documents)
- Score threshold filtering (default: 0.35)
- Manual knowledge base selection from UI

**5. Gateway Integration**
- User-selected knowledge base passed via `knowledge_base` request field
- Context injection into system prompt
- Graceful fallback if RAG unavailable or no KB selected

**6. Data Collection Scripts**
- **ArXiv Collector**: Downloads ML/DL papers (cs.LG, cs.AI)
- **PyTorch Docs Scraper**: Collects core API documentation
- **Index Builder**: Chunks, embeds, and stores in Qdrant

**7. UI Updates**
- Knowledge base radio selector in sidebar (ArXiv papers / PyTorch docs / Disabled)
- Selected KB is sent with every chat request

---

## What's Postponed (Future Work)

### 🔄 Advanced Retrieval Strategies
- **Hybrid Search**: BM25 + dense retrieval (requires more indexing)
- **Multi-query Retrieval**: Query expansion using LLM
- **Metadata Filtering**: Date ranges, categories, versions

### 🔄 Reranking
- **Cross-encoder Reranking**: Improve relevance of top results
- **LLM-based Reranking**: Score passages with LLM

### 🔄 Additional Data Sources
- **HuggingFace Documentation**: Transformers library docs
- **GitHub Repositories**: Curated code examples
- **Technical Blogs**: ML/DL blog posts
- **Conference Papers**: NeurIPS, ICML, ICLR (full PDFs)

### 🔄 Evaluation & Metrics
- Retrieval quality (Recall@k, nDCG@k)
- Answer groundedness (citations, hallucination detection)
- End-to-end quality (LLM-as-judge)
- Latency profiling

### 🔄 Advanced Features
- **Context window management**: Smart truncation
- **Query history**: Remember previous retrievals
- **Citation extraction**: Show source papers/docs inline

### 🔄 Optimization
- **Embedding caching**: Cache frequent queries
- **Batch retrieval**: Process multiple queries together
- **Quantized embeddings**: Reduce index size

---

## Setup Instructions

### Step 1: Update Dependencies

The dependencies are already added to `pyproject.toml` and `requirements-gateway.txt`. Install them:

```bash
# For local development (includes all dependencies)
pip install -e .

# Or with uv (faster)
uv pip install -e .
```

### Step 2: Update Environment Configuration

Copy the example environment file and adjust if needed:

```bash
cd infra/compose
cp .env.example .env
```

Make sure these RAG-related variables are set in `.env`:

```bash
# Qdrant
QDRANT_PORT=6333

# Gateway RAG Configuration
GATEWAY_QDRANT_HOST=qdrant
GATEWAY_QDRANT_PORT=6333
GATEWAY_RAG_ENABLED=true
GATEWAY_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Step 3: Start Infrastructure Services

Start Qdrant and other services:

```bash
cd infra/compose
docker-compose up -d postgres mlflow qdrant
```

Verify Qdrant is running:
```bash
curl http://localhost:6333/collections
# Should return: {"result":{"collections":[]}}
```

### Step 4: Collect Data

Data collection (ArXiv papers for chat RAG, PyTorch docs for code RAG) can be done in
two ways:

#### Option A: Automated via Airflow DAGs (recommended for production)

Start the full Docker Compose stack (including Airflow):

```bash
cd infra/compose
docker-compose up -d
```

Two Airflow DAGs manage data collection automatically:

| DAG | Schedule | What it does |
|-----|----------|-------------|
| `arxiv_rag_update` | Daily | Downloads latest ArXiv papers → DVC version → rebuilds `chat_documents` index |
| `pytorch_docs_rag_update` | Weekly | Scrapes PyTorch docs → DVC version → rebuilds `code_documents` index |

Open Airflow UI at `http://localhost:8080` (default login: `admin` / `admin`) to
monitor DAG runs, trigger manual runs, or review logs.

To trigger a DAG immediately (without waiting for the schedule):
1. Open Airflow UI → DAGs page
2. Toggle the DAG on (if paused)
3. Click the "Trigger DAG" button (▶)

#### Option B: Interactive via notebook (for development/exploration)

Open `experiments/scripts/prefetch_assets.ipynb`:

- **Section 8** — ArXiv papers (chat RAG)
- **Section 9** — PyTorch documentation (code RAG)

Set `PROJECT_ROOT` and run the relevant sections.

**Outputs:**
- `assets/rag_data/arxiv/arxiv_papers.json`
- `assets/rag_data/pytorch_docs/pytorch_docs.json`

**Time**: ~1-2 minutes (respects rate limits)

### Step 5: Build Vector Indices

```bash
# Build both chat and code indices
python build_vector_index.py \
    --task both \
    --qdrant-host localhost \
    --qdrant-port 6333 \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --force-recreate

# Or build separately:
# python build_vector_index.py --task chat
# python build_vector_index.py --task code
```

**Time**:
- Chat index (100 papers): ~3-5 minutes
- Code index (~15 docs): ~1 minute

**Output**:
- Qdrant collection: `chat_documents` (~300-400 chunks)
- Qdrant collection: `code_documents` (~100-150 chunks)

Verify indices were created:
```bash
curl http://localhost:6333/collections
# Should show: chat_documents, code_documents
```

Check collection stats:
```bash
curl http://localhost:6333/collections/chat_documents
curl http://localhost:6333/collections/code_documents
```

---

## Running the System

### Start All Services

```bash
cd infra/compose
docker-compose up -d
```

This starts:
- PostgreSQL (MLflow backend)
- MLflow
- Qdrant
- vLLM (with base model)
- Gateway (with RAG enabled)
- UI
- Airflow (scheduler + webserver — manages RAG data update DAGs)

### Check Service Health

```bash
# Gateway health
curl http://localhost:9000/health

# Qdrant health
curl http://localhost:6333/collections

# vLLM health
curl http://localhost:8000/v1/models
```

### Access UI

Open browser: http://localhost:8501

You should see:
- Main chat interface
- Sidebar with **Knowledge Base** selector (Disabled / ArXiv papers / PyTorch docs)
- Max tokens setting

---

## Testing RAG

### Test Query Examples

**Test 1: ArXiv Knowledge Base (ML/AI Theory)**

Select **"ArXiv papers (ML / AI theory)"** in the sidebar, then ask:

```
User: "What are the main approaches to fine-tuning large language models?"
```

Expected behavior:
- UI sends `knowledge_base: "arxiv"` with the request
- Retrieves from `chat_documents` collection
- Injects ArXiv paper abstracts into context
- LLM generates answer using retrieved context

**Test 2: PyTorch Docs Knowledge Base (Coding)**

Select **"PyTorch docs (coding)"** in the sidebar, then ask:

```
User: "How do I create a neural network in PyTorch?"
User: "Show me python code for a simple CNN"
```

Expected behavior:
- UI sends `knowledge_base: "pytorch_docs"` with the request
- Retrieves from `code_documents` collection
- Injects PyTorch documentation into context
- LLM generates code with proper API usage

**Test 3: No Knowledge Base (RAG Disabled)**

Select **"Disabled"** in the sidebar, then ask:

```
User: "Summarize the key concepts in attention mechanisms"
```

Expected behavior:
- No RAG retrieval is performed
- LLM answers using its own knowledge

### Viewing Retrieved Context

Check gateway logs to see RAG in action:

```bash
docker-compose logs -f gateway
```

Look for log messages like:
```
INFO: RAG — retrieving from knowledge base: arxiv
INFO: Retrieved 5 documents
INFO: RAG context retrieved (kb=arxiv)
```

### Testing Without Docker

If running gateway locally:

```bash
# Set environment variables
export GATEWAY_QDRANT_HOST=localhost
export GATEWAY_QDRANT_PORT=6333
export GATEWAY_RAG_ENABLED=true
export GATEWAY_VLLM_BASE_URL=http://localhost:8000

# Run gateway
cd src
python -m uvicorn gateway.main:app --host 0.0.0.0 --port 9000
```

---

## Troubleshooting

### Issue: "Collection does not exist"

**Problem**: Gateway logs show collections not found

**Solution**:
```bash
# Check if indices were built
curl http://localhost:6333/collections

# Rebuild indices
cd experiments/scripts/rag_data
python build_vector_index.py --task both --force-recreate
```

### Issue: Gateway crashes on startup

**Problem**: RAG initialization fails

**Solution**:
```bash
# Check Qdrant is running
docker-compose ps qdrant

# Check Qdrant logs
docker-compose logs qdrant

# Restart gateway
docker-compose restart gateway
```

### Issue: No context retrieved (empty results)

**Problem**: Queries return no documents

**Possible causes**:
1. No knowledge base selected in the UI sidebar
2. Query too different from indexed content
3. Score threshold too high
4. Selected collection has no documents

**Solution**:
- Make sure a knowledge base is selected in the sidebar (not "Disabled")
- Try lowering `GATEWAY_SCORE_THRESHOLD` (default: 0.35)
- Check collection has documents: `curl http://localhost:6333/collections/chat_documents`
- Try more specific queries related to ML/DL topics

### Issue: Slow embedding generation

**Problem**: First query takes long time

**Solution**:
- First query downloads embedding model (~80MB)
- Subsequent queries are fast (<100ms)
- Model is cached in `~/.cache/torch/sentence_transformers/`

### Issue: Out of memory on GPU

**Problem**: vLLM crashes with OOM error

**Solution**:
```bash
# Reduce vLLM GPU utilization in .env
VLLM_GPU_UTIL=0.6  # Instead of 0.7

# Verify embeddings run on CPU (check gateway logs)
# Should see: "Loading embedding model on device: cpu"
```

---

## Next Steps

After baseline RAG is working:

1. **Evaluate Quality**:
   - Create test queries dataset
   - Measure retrieval quality (Recall@k)
   - Test answer quality with LLM-as-judge

2. **Expand Data**:
   - Add more ArXiv papers (increase --max-results)
   - Add HuggingFace documentation
   - Add curated code examples

3. **Experiment with Chunking**:
   - Try different chunk sizes (256, 512, 1024)
   - Test section-aware chunking for papers
   - Measure impact on retrieval quality

4. **Add Hybrid Search**:
   - Implement BM25 for keyword matching
   - Combine dense + sparse retrieval
   - Compare performance

5. **Implement Reranking**:
   - Add cross-encoder reranking
   - Measure improvement in relevance

6. **Monitor Performance**:
   - Log retrieval latency
   - Track context length
   - Monitor GPU/CPU usage

---

## File Structure

```
agent-042/
├── src/
│   ├── rag/                          # RAG module
│   │   ├── config.py                # RAG settings
│   │   ├── embeddings.py            # Embedding service
│   │   ├── vector_store.py          # Qdrant client
│   │   ├── chunking.py              # Document chunking
│   │   └── retriever.py             # Retrieval orchestration
│   └── gateway/
│       └── services/
│           ├── rag_service.py       # Gateway RAG integration
│           ├── prompt_builder.py    # Context injection
│           └── processing.py        # RAG invocation
├── dags/                             # Airflow DAGs (data pipelines)
│   ├── arxiv_rag_update.py          # Daily: ArXiv download → DVC → index
│   ├── pytorch_docs_rag_update.py   # Weekly: PyTorch docs → DVC → index
├── experiments/scripts/
│   ├── prefetch_assets.ipynb        # Data collection (ArXiv, PyTorch docs, etc.)
│   └── rag_data/
│       └── build_vector_index.py    # Index building (used by DAGs)
├── assets/rag_data/
│   ├── arxiv/                       # ArXiv papers JSON
│   └── pytorch_docs/                # PyTorch docs JSON
├── infra/compose/
│   ├── docker-compose.yaml          # Full stack (incl. Qdrant, Airflow)
│   └── .env.example                 # RAG + Airflow config
├── infra/docker/airflow/
│   └── requirements.txt             # DAG Python dependencies
└── RAG-SETUP.md                     # This file
```

---

## Support

For issues or questions:
1. Check logs: `docker-compose logs gateway qdrant`
2. Verify collections: `curl http://localhost:6333/collections`
3. Review this document's Troubleshooting section
4. Check README.md for overall system architecture
