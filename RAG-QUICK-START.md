# RAG System - Quick Start Guide

**TL;DR**: Baseline RAG system for agent-042, optimized for RTX 3060 (12GB VRAM).

---

## Quick Setup (5 Commands)

```bash
# 1. Update .env file with RAG settings (already configured in .env.example)
cd infra/compose
cp .env.example .env

# 2. Start Qdrant
docker-compose up -d qdrant

# 3. Collect data (ArXiv papers + PyTorch docs)
cd ../../experiments/scripts/rag_data
python collect_arxiv.py --max-results 100 --output-dir ../../../assets/rag_data/arxiv
python collect_pytorch_docs.py --output-dir ../../../assets/rag_data/pytorch_docs

# 4. Build vector indices
python build_vector_index.py --task both --qdrant-host localhost --force-recreate

# 5. Start full system
cd ../../../infra/compose
docker-compose up -d
```

**Total time**: ~10-15 minutes

---

## Architecture Overview

```
┌─────────────┐      ┌──────────────┐      ┌────────────┐
│   Streamlit │─────▶│   Gateway    │─────▶│   vLLM     │
│     UI      │      │   (FastAPI)  │      │  (RTX3060) │
└─────────────┘      └──────┬───────┘      └────────────┘
                            │
                            │ retrieves context
                            ▼
                     ┌─────────────┐
                     │   Qdrant    │
                     │ Vector  DB  │
                     └─────────────┘
                            │
                     ┌──────┴──────┐
                  chat_docs    code_docs
                  (ArXiv)    (PyTorch)
```

---

## Key Components

| Component | Model/Tool | VRAM | Device |
|-----------|-----------|------|--------|
| **Base LLM** | Qwen3-0.6B (quantized) | 8-9GB | GPU |
| **Embeddings** | all-MiniLM-L6-v2 | 0GB | CPU |
| **Vector DB** | Qdrant | 0GB | Disk |
| **Total** | - | ~9GB | - |

**Remaining VRAM**: ~3GB buffer

---

## What's Included (Baseline)

✅ **Infrastructure**
- Qdrant vector database (Docker)
- Lightweight embedding model (CPU-based)
- 2 collections: chat (ArXiv), code (PyTorch)

✅ **Data**
- ~100 ArXiv papers (ML/DL categories)
- ~15 PyTorch documentation pages
- ~400-500 total chunks indexed

✅ **Retrieval**
- Dense vector search (cosine similarity)
- Top-5 retrieval per query
- Task-based routing (auto-detect)

✅ **Integration**
- Automatic context injection
- Gateway-level RAG service
- UI status indicator

---

## What's Postponed (Future)

⏳ **Advanced Retrieval**
- Hybrid search (BM25 + dense)
- Multi-query expansion
- Cross-encoder reranking

⏳ **More Data**
- HuggingFace docs
- GitHub repositories
- Technical blogs
- Full paper PDFs

⏳ **Evaluation**
- Retrieval metrics (Recall@k, nDCG@k)
- Answer quality (LLM-as-judge)
- Latency profiling

⏳ **Features**
- Per-query RAG toggle
- Citation extraction
- Query history
- Embedding caching

---

## Testing RAG

**Chat Query (ArXiv)**:
```
"What are the main approaches to fine-tuning LLMs?"
→ Retrieves from chat_documents (ArXiv papers)
```

**Code Query (PyTorch)**:
```
"Show me Python code for a CNN"
→ Retrieves from code_documents (PyTorch docs)
```

**Check Logs**:
```bash
docker-compose logs -f gateway | grep RAG
# Should see: "RAG context retrieved for task: chat"
```

---

## Common Issues

### Collections not found
```bash
curl http://localhost:6333/collections
# If empty, rebuild: python build_vector_index.py --task both --force-recreate
```

### No context retrieved
- Lower score threshold in `src/rag/config.py` (0.3 → 0.1)
- Try queries about ML/DL topics (closer to ArXiv content)

### OOM on GPU
- Reduce `VLLM_GPU_UTIL` in `.env` (0.7 → 0.6)
- Verify embeddings on CPU: check gateway logs

---

## Files Changed

**New Files**:
- `src/rag/` - RAG module (embeddings, vector store, chunking, retrieval)
- `src/gateway/services/rag_service.py` - Gateway integration
- `experiments/scripts/rag_data/` - Data collection scripts
- `RAG-SETUP.md` - Full documentation (this file)

**Modified Files**:
- `pyproject.toml` - Added RAG dependencies
- `requirements-gateway.txt` - Added RAG dependencies
- `infra/compose/docker-compose.yaml` - Added Qdrant service
- `infra/compose/.env.example` - Added RAG config
- `src/gateway/config.py` - Added RAG settings
- `src/gateway/services/prompt_builder.py` - Context injection
- `src/gateway/services/processing.py` - RAG invocation
- `src/ui/app.py` - RAG status display

---

## Next Steps

1. **Test basic RAG**: Ask ML/DL questions, check logs
2. **Evaluate quality**: Create test queries, measure accuracy
3. **Expand data**: More ArXiv papers, add HuggingFace docs
4. **Optimize**: Experiment with chunk sizes, retrieval parameters
5. **Add features**: Hybrid search, reranking, citations

---

## Full Documentation

See `RAG-SETUP.md` for:
- Detailed architecture
- Step-by-step setup
- Comprehensive troubleshooting
- Future roadmap
- File structure

---

## Support

**Verify Setup**:
```bash
# Qdrant
curl http://localhost:6333/collections

# Gateway health
curl http://localhost:9000/health

# Check logs
docker-compose logs gateway qdrant
```

**Data Location**:
- Raw data: `assets/rag_data/`
- Vector indices: Qdrant volume (persistent)

**Rebuild Indices**:
```bash
cd experiments/scripts/rag_data
python build_vector_index.py --task both --force-recreate
```
