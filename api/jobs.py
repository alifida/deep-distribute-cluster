"""
API routes for job management.

Endpoints:
- POST /jobs -> create job
- GET /jobs -> list jobs
- GET /jobs/{job_id} -> get job
- PATCH /jobs/{job_id} -> update job

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import uuid

from utils.db import get_db
from models.db_models import Job
from schemas.schemas import JobCreate, JobUpdate, JobOut

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("/", response_model=JobOut)
def create_job(job_in: JobCreate, db: Session = Depends(get_db)):
    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        model=job_in.model,
        dataset=job_in.dataset,
        status="pending",
        progress=0.0,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


@router.get("/", response_model=list[JobOut])
def list_jobs(db: Session = Depends(get_db)):
    return db.query(Job).all()


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.patch("/{job_id}", response_model=JobOut)
def update_job(job_id: str, job_in: JobUpdate, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job_in.status is not None:
        job.status = job_in.status
    if job_in.progress is not None:
        job.progress = job_in.progress

    db.commit()
    db.refresh(job)
    return job

"""