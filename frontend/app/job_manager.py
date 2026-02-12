"""
Async job manager for background document processing.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Optional
from enum import Enum
from dataclasses import dataclass, field


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    filename: str
    status: JobStatus = JobStatus.PENDING
    progress: int = 0
    message: str = "Queued"
    doc_id: Optional[str] = None
    doc_name: Optional[str] = None
    chunks: int = 0
    pages: int = 0
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class JobManager:
    """Manages background processing jobs."""
    
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()
    
    async def create_job(self, filename: str) -> str:
        """Create a new job and return its ID."""
        job_id = str(uuid.uuid4())[:8]
        async with self._lock:
            self._jobs[job_id] = Job(id=job_id, filename=filename)
        return job_id
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        async with self._lock:
            return self._jobs.get(job_id)
    
    async def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        doc_id: Optional[str] = None,
        doc_name: Optional[str] = None,
        chunks: Optional[int] = None,
        pages: Optional[int] = None,
        error: Optional[str] = None
    ):
        """Update job status."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                if status:
                    job.status = status
                if progress is not None:
                    job.progress = progress
                if message:
                    job.message = message
                if doc_id:
                    job.doc_id = doc_id
                if doc_name:
                    job.doc_name = doc_name
                if chunks is not None:
                    job.chunks = chunks
                if pages is not None:
                    job.pages = pages
                if error:
                    job.error = error
                if status == JobStatus.COMPLETED or status == JobStatus.FAILED:
                    job.completed_at = datetime.now()
    
    async def get_all_jobs(self) -> Dict[str, Job]:
        """Get all jobs."""
        async with self._lock:
            return dict(self._jobs)
    
    async def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Remove jobs older than max_age_hours."""
        async with self._lock:
            now = datetime.now()
            to_remove = []
            for job_id, job in self._jobs.items():
                if job.completed_at:
                    age = (now - job.completed_at).total_seconds() / 3600
                    if age > max_age_hours:
                        to_remove.append(job_id)
            for job_id in to_remove:
                del self._jobs[job_id]


# Global job manager instance
job_manager = JobManager()
