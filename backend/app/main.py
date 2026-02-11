"""
Simple RAG Chatbot API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import ingestion, chat

app = FastAPI(title="RAG Chatbot", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingestion.router)
app.include_router(chat.router)


@app.get("/")
async def home():
    return {
        "message": "RAG Chatbot API",
        "usage": {
            "1. Upload PDF": "POST /ingestion/upload",
            "2. Ask questions": "POST /chat/ask",
            "3. Check status": "GET /ingestion/status"
        }
    }
