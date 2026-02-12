"""
Async embeddings using Voyage AI with batching, retries, and progress updates.
"""

import time
import asyncio
import logging
from typing import List, Optional
from voyageai import Client as VoyageClient
from app.config import settings

logger = logging.getLogger(__name__)

# Initialize client with longer timeout
voyage = VoyageClient(
    api_key=settings.VOYAGE_API_KEY,
    timeout=180  # 3 minute timeout
)

BATCH_SIZE = 10  # Larger batches = fewer API calls
BATCH_DELAY = 1  # Minimal delay between successful batches
MAX_RETRIES = 5  # Retries for rate limits


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed multiple texts with batching (sync version)."""
    all_embeddings = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        for attempt in range(MAX_RETRIES):
            try:
                result = voyage.embed(batch, model="voyage-3", input_type="document")
                all_embeddings.extend(result.embeddings)
                logger.info(f"Batch {batch_num}/{total_batches} done")
                break
            except Exception as e:
                wait_time = 20 * (attempt + 1)  # 20, 40, 60, 80, 100 seconds
                logger.warning(f"Error on batch {batch_num}, waiting {wait_time}s: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Voyage AI failed after {MAX_RETRIES} retries: {e}")
        
        if i + BATCH_SIZE < len(texts):
            time.sleep(BATCH_DELAY)
    
    return all_embeddings


async def embed_texts_async(texts: List[str], job_id: Optional[str] = None) -> List[List[float]]:
    """Async embed with progress updates to job manager."""
    from app.job_manager import job_manager
    
    all_embeddings = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        # Update progress
        if job_id:
            progress = 50 + int((batch_num / total_batches) * 45)  # 50-95%
            await job_manager.update_job(
                job_id,
                progress=progress,
                message=f"Embedding batch {batch_num}/{total_batches}..."
            )
        
        # Retry with longer backoff for connection issues
        for attempt in range(MAX_RETRIES):
            try:
                # Run in thread pool to not block
                result = await asyncio.to_thread(
                    voyage.embed, batch, model="voyage-3", input_type="document"
                )
                all_embeddings.extend(result.embeddings)
                break
            except Exception as e:
                wait_time = 20 * (attempt + 1)  # 20, 40, 60, 80, 100 seconds
                logger.warning(f"Error on batch {batch_num}, waiting {wait_time}s (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"Voyage AI failed after {MAX_RETRIES} retries: {e}")
        
        # Delay between batches (non-blocking)
        if i + BATCH_SIZE < len(texts):
            await asyncio.sleep(BATCH_DELAY)
    
    return all_embeddings


def embed_query(text: str) -> List[float]:
    """Embed a query for searching with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            result = voyage.embed([text], model="voyage-3", input_type="query")
            return result.embeddings[0]
        except Exception as e:
            wait_time = 20 * (attempt + 1)
            logger.warning(f"Query embed error, waiting {wait_time}s: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait_time)
            else:
                raise Exception(f"Voyage AI query embed failed: {e}")
