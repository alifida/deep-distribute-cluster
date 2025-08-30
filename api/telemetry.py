"""
API routes for telemetry logging and retrieval.

Endpoints:
- POST /telemetry -> log telemetry
- GET /telemetry/job/{job_id} -> get telemetry logs for a job
- GET /telemetry/worker/{worker_id} -> get telemetry logs for a worker

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from utils.db import get_db
from models.db_models import TelemetryLog
from schemas.schemas import TelemetryLogCreate, TelemetryLogOut

router = APIRouter(prefix="/telemetry", tags=["telemetry"])


@router.post("/", response_model=TelemetryLogOut)
def log_telemetry(log_in: TelemetryLogCreate, db: Session = Depends(get_db)):
    log = TelemetryLog(
        job_id=log_in.job_id,
        worker_id=log_in.worker_id,
        metric=log_in.metric,
        value=log_in.value,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


@router.get("/job/{job_id}", response_model=list[TelemetryLogOut])
def get_job_telemetry(job_id: str, db: Session = Depends(get_db)):
    return db.query(TelemetryLog).filter(TelemetryLog.job_id == job_id).all()


@router.get("/worker/{worker_id}", response_model=list[TelemetryLogOut])
def get_worker_telemetry(worker_id: str, db: Session = Depends(get_db)):
    return db.query(TelemetryLog).filter(TelemetryLog.worker_id == worker_id).all()

"""