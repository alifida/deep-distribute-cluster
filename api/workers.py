"""
API routes for worker management.

Endpoints:
- POST /workers -> register worker
- GET /workers -> list workers
- PATCH /workers/{worker_id}/heartbeat -> update heartbeat

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import time

from utils.db import get_db
from models.db_models import Worker
from schemas.schemas import WorkerRegister, WorkerOut

router = APIRouter(prefix="/workers", tags=["workers"])


@router.post("/", response_model=WorkerOut)
def register_worker(worker_in: WorkerRegister, db: Session = Depends(get_db)):
    worker = db.query(Worker).filter(Worker.id == worker_in.id).first()
    if worker:
        raise HTTPException(status_code=400, detail="Worker already registered")

    worker = Worker(id=worker_in.id, status="idle", last_heartbeat=time.time())
    db.add(worker)
    db.commit()
    db.refresh(worker)
    return worker


@router.get("/", response_model=list[WorkerOut])
def list_workers(db: Session = Depends(get_db)):
    return db.query(Worker).all()


@router.patch("/{worker_id}/heartbeat", response_model=WorkerOut)
def update_heartbeat(worker_id: str, db: Session = Depends(get_db)):
    worker = db.query(Worker).filter(Worker.id == worker_id).first()
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    worker.last_heartbeat = time.time()
    worker.status = "active"
    db.commit()
    db.refresh(worker)
    return worker

"""