"""
Parameter Server (PS) module.

Exposes FastAPI router with endpoints for:
- job submission & status tracking
- worker registration & heartbeats
- receiving worker weight updates
- aggregating model parameters
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import uuid
import time

from api.schemas import (
    TrainJobRequest,
    TrainJobResponse,
    JobStatusResponse,
    WorkerRegistration,
    WorkerHeartbeat,
    WeightUpdate,
    AggregatedWeights,
)

router = APIRouter()

# In-memory stores (replace with DB later)
JOBS: Dict[str, Dict[str, Any]] = {}
WORKERS: Dict[str, Dict[str, Any]] = {}
WEIGHTS: Dict[str, Dict[int, list]] = {}


# ----------------------------
# Job management
# ----------------------------
@router.post("/submit", response_model=TrainJobResponse)
def submit_job(req: TrainJobRequest):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "submitted": time.time(),
        "params": req.dict(),
        "metrics": {},
    }
    return TrainJobResponse(job_id=job_id, status="queued", message="Job submitted")


@router.get("/status/{job_id}", response_model=JobStatusResponse)
def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, detail="Job not found")
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        metrics=job.get("metrics"),
    )


# ----------------------------
# Worker management
# ----------------------------
@router.post("/register")
def register_worker(req: WorkerRegistration):
    WORKERS[req.worker_id] = {"host": req.host, "capabilities": req.capabilities, "last_seen": time.time()}
    return {"status": "registered", "worker_id": req.worker_id}


@router.post("/heartbeat")
def worker_heartbeat(req: WorkerHeartbeat):
    if req.worker_id not in WORKERS:
        raise HTTPException(404, detail="Worker not registered")
    WORKERS[req.worker_id]["last_seen"] = time.time()
    return {"status": "alive"}


# ----------------------------
# Weight synchronization
# ----------------------------
@router.post("/weights/update")
def update_weights(req: WeightUpdate):
    if req.job_id not in WEIGHTS:
        WEIGHTS[req.job_id] = {}
    WEIGHTS[req.job_id][req.step] = WEIGHTS[req.job_id].get(req.step, []) + req.weights
    return {"status": "received"}


@router.get("/weights/aggregate/{job_id}/{step}", response_model=AggregatedWeights)
def aggregate_weights(job_id: str, step: int):
    if job_id not in WEIGHTS or step not in WEIGHTS[job_id]:
        raise HTTPException(404, detail="No weights found")

    weights = WEIGHTS[job_id][step]
    if not weights:
        raise HTTPException(404, detail="Empty weights")

    # simple average aggregation
    n = len(weights)
    avg = sum(weights) / n

    return AggregatedWeights(job_id=job_id, step=step, weights=[avg])
