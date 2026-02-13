"""
Vector store with multi-document support using Qdrant.
"""

import uuid
import asyncio
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from app.config import settings
from app.embeddings import embed_texts, embed_query, embed_texts_async

# Initialize Qdrant client
qdrant = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)


def setup_collection():
    """Create collection if it doesn't exist and ensure payload index."""
    from qdrant_client.models import PayloadSchemaType
    collections = [c.name for c in qdrant.get_collections().collections]
    if settings.COLLECTION_NAME not in collections:
        qdrant.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
    # Always try to create payload index (idempotent operation)
    try:
        qdrant.create_payload_index(
            collection_name=settings.COLLECTION_NAME,
            field_name="doc_id",
            field_schema=PayloadSchemaType.KEYWORD
        )
    except:
        pass  # Index already exists


def add_chunks(chunks: List[Dict], doc_id: str, doc_name: str) -> int:
    """Add text chunks with document ID for filtering."""
    setup_collection()
    
    texts = [c["content"] for c in chunks]
    embeddings = embed_texts(texts)
    
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "content": chunk["content"],
                "source": chunk.get("source", ""),
                "doc_id": doc_id,
                "doc_name": doc_name
            }
        )
        for chunk, emb in zip(chunks, embeddings)
    ]
    
    qdrant.upsert(collection_name=settings.COLLECTION_NAME, points=points)
    return len(points)


async def add_chunks_async(chunks: List[Dict], doc_id: str, doc_name: str, job_id: str = None) -> int:
    """Async version: Add text chunks with progress updates."""
    from app.job_manager import job_manager
    
    setup_collection()
    
    texts = [c["content"] for c in chunks]
    
    # Use async embedding with progress updates
    embeddings = await embed_texts_async(texts, job_id)
    
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "content": chunk["content"],
                "source": chunk.get("source", ""),
                "doc_id": doc_id,
                "doc_name": doc_name
            }
        )
        for chunk, emb in zip(chunks, embeddings)
    ]
    
    # Upsert in thread pool to not block
    await asyncio.to_thread(
        qdrant.upsert,
        collection_name=settings.COLLECTION_NAME,
        points=points
    )
    
    return len(points)


def search(query: str, doc_id: Optional[str] = None, limit: int = 5) -> List[Dict]:
    """Search for relevant chunks, optionally filtered by document ID."""
    setup_collection()  # Ensure index exists
    query_vector = embed_query(query)
    
    # Build filter if doc_id provided
    query_filter = None
    if doc_id:
        query_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )
    
    results = qdrant.query_points(
        collection_name=settings.COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=limit
    )
    
    return [
        {
            "content": r.payload["content"],
            "score": r.score,
            "doc_id": r.payload.get("doc_id"),
            "doc_name": r.payload.get("doc_name")
        }
        for r in results.points
    ]


def get_documents() -> List[Dict]:
    """Get list of all unique documents."""
    try:
        # Scroll through all points to get unique doc_ids
        all_points, _ = qdrant.scroll(
            collection_name=settings.COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        
        # Extract unique documents
        docs = {}
        for point in all_points:
            doc_id = point.payload.get("doc_id")
            doc_name = point.payload.get("doc_name", "Unknown")
            if doc_id and doc_id not in docs:
                docs[doc_id] = {"id": doc_id, "name": doc_name, "chunks": 0}
            if doc_id:
                docs[doc_id]["chunks"] += 1
        
        return list(docs.values())
    except:
        return []


def get_stats() -> Dict:
    """Get collection statistics."""
    try:
        info = qdrant.get_collection(settings.COLLECTION_NAME)
        docs = get_documents()
        return {
            "total_chunks": info.points_count,
            "documents": len(docs),
            "document_list": docs,
            "status": "ready" if info.points_count > 0 else "empty"
        }
    except:
        return {"total_chunks": 0, "documents": 0, "document_list": [], "status": "empty"}


def delete_document(doc_id: str) -> bool:
    """Delete all chunks for a specific document."""
    try:
        qdrant.delete(
            collection_name=settings.COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )
        )
        return True
    except:
        return False


def clear():
    """Delete all documents."""
    try:
        qdrant.delete_collection(settings.COLLECTION_NAME)
    except:
        pass
