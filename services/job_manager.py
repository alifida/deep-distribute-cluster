"""
Job Lifecycle Manager

Handles job status transitions and persistence.
- Create job
- Assign job to worker
- Update status (pending -> running -> completed/failed)
- Store metrics & results
"""
"""
from sqlalchemy.orm import Session
from models.job import Job
from models.worker import Worker
from utils.logger import get_logger

logger = get_logger("job-manager")


class JobManager:
    def __init__(self, db: Session):
        self.db = db

    def create_job(self, job_data: dict) -> Job:
        job = Job(**job_data)
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        logger.info(f"Created job {job.id}")
        return job

    def assign_job(self, job_id: int, worker_id: int) -> Job:
        job = self.db.query(Job).filter(Job.id == job_id).first()
        worker = self.db.query(Worker).filter(Worker.id == worker_id).first()
        if not job or not worker:
            raise ValueError("Invalid job or worker ID")
        job.worker_id = worker_id
        job.status = "running"
        self.db.commit()
        self.db.refresh(job)
        logger.info(f"Assigned job {job_id} to worker {worker_id}")
        return job

    def complete_job(self, job_id: int, success: bool = True, metrics: dict = None):
        job = self.db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise ValueError("Invalid job ID")
        job.status = "completed" if success else "failed"
        if metrics:
            job.metrics = metrics
        self.db.commit()
        self.db.refresh(job)
        logger.info(f"Job {job_id} marked as {job.status}")
        return job

    def get_job(self, job_id: int) -> Job:
        job = self.db.query(Job).filter(Job.id == job_id).first()
        return job

    def list_jobs(self, status: str = None):
        query = self.db.query(Job)
        if status:
            query = query.filter(Job.status == status)
        return query.all()

"""