"""
Chat router with document selection and history support.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

from app.vector_store import search
from app.llm_service import ask_with_history

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None  # Filter by specific document
    history: Optional[List[ChatMessage]] = None


@router.post("/ask")
async def ask_question(req: ChatRequest):
    """Ask a question about a specific document or all documents."""
    try:
        # Find relevant chunks (filtered by doc_id if provided)
        results = search(req.question, doc_id=req.doc_id, limit=5)
        
        if not results:
            if req.doc_id:
                return {"answer": "No content found in this document.", "sources": [], "doc_id": req.doc_id}
            return {"answer": "No documents found. Please upload a PDF first.", "sources": []}
        
        # Build context from search results
        context = "\n\n".join([r["content"] for r in results])
        
        # Convert history to list of dicts
        history = []
        if req.history:
            history = [{"role": m.role, "content": m.content} for m in req.history]
        
        # Get answer from LLM with history
        answer = ask_with_history(req.question, context, history)
        
        return {
            "answer": answer,
            "sources": [r["content"][:200] + "..." for r in results],
            "doc_id": req.doc_id
        }
    except Exception as e:
        import traceback
        print(f"CHAT ERROR: {traceback.format_exc()}")
        return {"answer": f"Error: {str(e)}", "sources": [], "doc_id": req.doc_id}
