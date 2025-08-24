"""
Telemetry module.

Provides endpoints for:
- workers to push logs and metrics
- admins to query telemetry data
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import time

from api.schemas import TelemetryLog

router = APIRouter()

# In-memory telemetry store (later to be backed by DB or message queue)
TELEMETRY_LOGS: List[Dict[str, Any]] = []


# ----------------------------
# Worker pushes telemetry
# ----------------------------
@router.post("/push")
def push_log(req: TelemetryLog):
    entry = {
        "worker_id": req.worker_id,
        "job_id": req.job_id,
        "timestamp": time.time(),
        "level": req.level,
        "message": req.message,
        "metrics": req.metrics or {},
    }
    TELEMETRY_LOGS.append(entry)
    return {"status": "logged"}


# ----------------------------
# Admin queries telemetry
# ----------------------------
@router.get("/logs")
def get_logs(limit: int = 50):
    return TELEMETRY_LOGS[-limit:]


@router.get("/logs/{job_id}")
def get_logs_for_job(job_id: str, limit: int = 50):
    logs = [log for log in TELEMETRY_LOGS if log["job_id"] == job_id]
    if not logs:
        raise HTTPException(404, detail="No logs found for this job")
    return logs[-limit:]
