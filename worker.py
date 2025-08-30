"""
Worker module.

Exposes FastAPI router with endpoints for:
- fetching assigned jobs from PS
- reporting training progress & metrics
- sending weight updates
- heartbeat back to PS
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import time

from api.schemas import (
    TrainJobResponse,
    JobStatusResponse,
    WorkerHeartbeat,
    WeightUpdate,
)

router = APIRouter()

# In-memory state (to be coordinated with PS later)
ASSIGNED_JOBS: Dict[str, Dict[str, Any]] = {}


# ----------------------------
# Job fetch & status
# ----------------------------
@router.get("/job/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    job = ASSIGNED_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, detail="Job not found on worker")
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        metrics=job.get("metrics"),
    )


@router.post("/job/{job_id}/start", response_model=TrainJobResponse)
def start_job(job_id: str):
    if job_id not in ASSIGNED_JOBS:
        ASSIGNED_JOBS[job_id] = {"status": "running", "progress": 0.0, "metrics": {}, "started": time.time()}
    else:
        ASSIGNED_JOBS[job_id]["status"] = "running"
        ASSIGNED_JOBS[job_id]["started"] = time.time()
    return TrainJobResponse(job_id=job_id, status="running", message="Job started on worker")


# ----------------------------
# Progress & heartbeat
# ----------------------------
@router.post("/heartbeat")
def heartbeat(req: WorkerHeartbeat):
    return {"status": "alive", "worker_id": req.worker_id}


@router.post("/job/{job_id}/progress")
def update_progress(job_id: str, progress: float, metrics: Dict[str, Any] = None):
    if job_id not in ASSIGNED_JOBS:
        raise HTTPException(404, detail="Job not found on worker")
    ASSIGNED_JOBS[job_id]["progress"] = progress
    if metrics:
        ASSIGNED_JOBS[job_id]["metrics"] = metrics
    return {"status": "updated", "progress": progress}


# ----------------------------
# Weight synchronization
# ----------------------------
@router.post("/weights/send")
def send_weights(req: WeightUpdate):
    # In real implementation, this would forward to PS
    return {"status": "sent", "job_id": req.job_id, "step": req.step}
