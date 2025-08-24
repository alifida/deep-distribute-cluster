"""
Job API Routes

Endpoints:
- POST /jobs/submit
- GET /jobs/{job_id}/telemetry
- POST /jobs/{job_id}/finalize

from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session
from utils.db import get_db
from services.orchestrator import Orchestrator

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("/submit")
def submit_job(
    dataset_id: int = Form(...),
    model_name: str = Form(...),
    version: str = Form(...),
    description: str = Form(None),
    db: Session = Depends(get_db),
):
    orchestrator = Orchestrator(db)
    job = orchestrator.submit_job(dataset_id, model_name, version, description)
    return {"job_id": job.id, "status": job.status}


@router.get("/{job_id}/telemetry")
def get_telemetry(job_id: int, db: Session = Depends(get_db)):
    orchestrator = Orchestrator(db)
    agg = orchestrator.collect_telemetry(job_id)
    return {"job_id": job_id, "aggregated_metrics": agg}


@router.post("/{job_id}/finalize")
def finalize_job(
    job_id: int,
    model_name: str = Form(...),
    version: str = Form(...),
    description: str = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # Save uploaded file temporarily
    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as f:
        f.write(file.file.read())

    orchestrator = Orchestrator(db)
    model = orchestrator.finalize_job(job_id, tmp_path, model_name, version, description)

    return {"model_id": model.id, "path": model.path}

"""