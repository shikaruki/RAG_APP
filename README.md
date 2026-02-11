# 🤖 RAG Chatbot with Multi-Document Support

A production-ready RAG (Retrieval-Augmented Generation) chatbot built with FastAPI and Streamlit. Upload multiple PDF documents and chat with them using AI.

## ✨ Features

- **Multi-Document Support** - Upload and manage multiple PDFs
- **Document Filtering** - Chat with specific documents or all at once
- **Async Ingestion** - Background processing with progress tracking
- **Chat History** - Context-aware conversations
- **Beautiful UI** - Modern Streamlit interface
- **Production Ready** - Deploy to Render with one click

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI |
| Frontend | Streamlit |
| Embeddings | Voyage AI (voyage-3, 1024-dim) |
| LLM | Groq (Llama 3.3 70B) |
| Vector Store | Qdrant Cloud |
| PDF Processing | PyPDF + LangChain |

## 📁 Project Structure

```
├── app/
│   ├── main.py           # FastAPI application
│   ├── config.py         # Configuration (env vars)
│   ├── embeddings.py     # Voyage AI embeddings with retry
│   ├── vector_store.py   # Qdrant integration
│   ├── llm_service.py    # Groq LLM service
│   ├── job_manager.py    # Async job tracking
│   └── routers/
│       ├── ingestion.py  # Document upload (async)
│       └── chat.py       # Chat endpoints
├── frontend.py           # Streamlit UI
├── .streamlit/config.toml # Streamlit config (30MB limit)
├── requirements.txt
├── render.yaml           # Render deployment config
└── .env.example
```

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone <your-repo-url>
cd SBI_Card_RAG
pip install -r requirements.txt
```

### 2. Get API Keys (Free Tiers Available)

| Service | Get Key | Free Tier |
|---------|---------|-----------|
| Voyage AI | https://dash.voyageai.com | ✅ Yes |
| Groq | https://console.groq.com | ✅ Yes |
| Qdrant Cloud | https://cloud.qdrant.io | ✅ Yes |

### 3. Configure Environment
```bash
cp .env.example .env
```

Edit `.env`:
```env
VOYAGE_API_KEY=your_voyage_key
GROQ_API_KEY=your_groq_key
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_key
```

### 4. Run Locally
```bash
# Terminal 1 - Backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 - Frontend
streamlit run frontend.py --server.port 8501
```

Open http://localhost:8501 in your browser.

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingestion/upload` | Upload PDF (returns job_id) |
| `GET` | `/ingestion/job/{job_id}` | Check ingestion progress |
| `GET` | `/ingestion/documents` | List all documents |
| `GET` | `/ingestion/status` | Get vector store stats |
| `DELETE` | `/ingestion/documents/{doc_id}` | Delete a document |
| `DELETE` | `/ingestion/clear` | Clear all documents |
| `POST` | `/chat/ask` | Ask question (with optional doc_id) |

### Example: Ask a Question
```bash
curl -X POST "http://localhost:8000/chat/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the revenue?", "doc_id": "abc123"}'
```

## ☁️ Deploy to Render

### 1. Push to GitHub
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

### 2. Create Services on Render
1. Go to https://render.com
2. Click **New** → **Blueprint**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml`

### 3. Add Environment Variables
For **sbi-card-rag-api** (backend):
- `VOYAGE_API_KEY`
- `GROQ_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`

For **sbi-card-rag-frontend** (frontend):
- `API_URL` = `https://sbi-card-rag-api.onrender.com`

### 4. Deploy! 🎉

## 📋 Configuration

### Chunking Settings (in `config.py`)
| Setting | Default | Description |
|---------|---------|-------------|
| CHUNK_SIZE | 3000 | Characters per chunk |
| CHUNK_OVERLAP | 200 | Overlap between chunks |
| MAX_CHUNKS | 50 | Max chunks per document |
| TOP_K | 5 | Results returned per query |

### Upload Limits
- **Max file size**: 30MB (configured in `.streamlit/config.toml`)

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Rate limit errors | Reduce batch size in `embeddings.py` |
| Timeout on upload | Increase timeout in `embeddings.py` |
| 500 error on chat | Check Qdrant connection/API key |
| Empty responses | Verify documents are indexed |

## 📄 License

MIT License

---

Built with ❤️ using FastAPI, Streamlit, and AI
