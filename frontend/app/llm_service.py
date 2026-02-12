"""
Simple LLM service using Groq with chat history.
"""

from typing import List, Dict
from groq import Groq
from app.config import settings

# Initialize client
groq = Groq(api_key=settings.GROQ_API_KEY)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
If you don't know the answer from the context, say so honestly.
Keep answers clear and concise. Use the chat history for context in follow-up questions."""


def ask_with_history(question: str, context: str, history: List[Dict] = None) -> str:
    """Ask a question with context and chat history."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add chat history (last 6 messages to stay within limits)
    if history:
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current question with context
    messages.append({
        "role": "user", 
        "content": f"Context from documents:\n{context}\n\nQuestion: {question}"
    })
    
    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return response.choices[0].message.content
