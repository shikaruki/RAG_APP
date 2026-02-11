"""
Async document ingestion router with background job processing.
"""

import os
import uuid
import asyncio
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from app.config import settings
from app.vector_store import add_chunks_async, get_stats, get_documents, delete_document, clear
from app.job_manager import job_manager, JobStatus

router = APIRouter(prefix="/ingestion", tags=["Ingestion"])


MAX_FILE_SIZE = 30 * 1024 * 1024  # 30 MB


def extract_text_from_pdf(pdf_path: str) -> tuple[str, int]:
    """Extract text from PDF using PyPDFLoader."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    page_count = len(pages)
    full_text = "\n\n".join([page.page_content for page in pages if page.page_content.strip()])
    
    return full_text, page_count


async def process_pdf_job(job_id: str, file_content: bytes, filename: str, tmp_path: str):
    """Background task to process PDF and add to vector store."""
    try:
        # Update status: extracting text
        await job_manager.update_job(
            job_id, 
            status=JobStatus.PROCESSING, 
            progress=10, 
            message="Extracting text from PDF..."
        )
        
        # Extract text (run in thread pool to not block)
        loop = asyncio.get_event_loop()
        full_text, page_count = await loop.run_in_executor(
            None, extract_text_from_pdf, tmp_path
        )
        
        await job_manager.update_job(job_id, progress=30, message="Text extracted", pages=page_count)
        
        if not full_text.strip():
            await job_manager.update_job(
                job_id, 
                status=JobStatus.FAILED, 
                error="No text content found in PDF"
            )
            return
        
        # Split into chunks
        await job_manager.update_job(job_id, progress=40, message="Splitting into chunks...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", ", ", " "],
            length_function=len
        )
        text_chunks = splitter.split_text(full_text)
        
        # Filter and prepare chunks
        chunks = [
            {"content": chunk.strip(), "source": filename}
            for chunk in text_chunks
            if chunk.strip() and len(chunk.strip()) > 50
        ]
        
        if not chunks:
            await job_manager.update_job(
                job_id, 
                status=JobStatus.FAILED, 
                error="No valid text chunks extracted"
            )
            return
        
        # Limit chunks
        if len(chunks) > settings.MAX_CHUNKS:
            chunks = chunks[:settings.MAX_CHUNKS]
        
        await job_manager.update_job(
            job_id, 
            progress=50, 
            message=f"Embedding {len(chunks)} chunks..."
        )
        
        # Generate document ID
        doc_id = str(uuid.uuid4())[:8]
        
        # Add to vector store (async)
        count = await add_chunks_async(chunks, doc_id, filename, job_id)
        
        # Mark complete
        await job_manager.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            message="Ingestion complete",
            doc_id=doc_id,
            doc_name=filename,
            chunks=count
        )
        
    except Exception as e:
        await job_manager.update_job(
            job_id,
            status=JobStatus.FAILED,
            error=str(e)
        )
    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except:
            pass


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Upload a PDF and start background processing. Returns job ID immediately."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    
    # Read and check file size
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large ({file_size_mb:.1f}MB). Maximum is 30MB.")
    
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    # Create job
    job_id = await job_manager.create_job(file.filename)
    
    # Start background processing
    asyncio.create_task(process_pdf_job(job_id, content, file.filename, tmp_path))
    
    return {
        "status": "processing",
        "job_id": job_id,
        "filename": file.filename,
        "size_mb": round(file_size_mb, 2),
        "message": "Upload accepted. Use job_id to check status."
    }


@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of an ingestion job."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    
    return {
        "job_id": job.id,
        "filename": job.filename,
        "status": job.status.value,
        "progress": job.progress,
        "message": job.message,
        "doc_id": job.doc_id,
        "doc_name": job.doc_name,
        "chunks": job.chunks,
        "pages": job.pages,
        "error": job.error
    }


@router.get("/jobs")
async def list_jobs():
    """List all jobs."""
    jobs = await job_manager.get_all_jobs()
    return {
        "jobs": [
            {
                "job_id": j.id,
                "filename": j.filename,
                "status": j.status.value,
                "progress": j.progress,
                "doc_id": j.doc_id
            }
            for j in jobs.values()
        ]
    }


@router.get("/documents")
async def list_documents():
    """Get list of all uploaded documents."""
    docs = await asyncio.to_thread(get_documents)
    return {"documents": docs, "count": len(docs)}


@router.delete("/documents/{doc_id}")
async def remove_document(doc_id: str):
    """Delete a specific document by ID."""
    success = await asyncio.to_thread(delete_document, doc_id)
    if success:
        return {"status": "deleted", "doc_id": doc_id}
    raise HTTPException(404, "Document not found")


@router.get("/status")
async def status():
    """Get the status of the knowledge base."""
    return get_stats()


@router.delete("/clear")
async def clear_all():
    """Clear all documents from the knowledge base."""
    clear()
    return {"status": "cleared"}
