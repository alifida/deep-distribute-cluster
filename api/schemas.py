"""
Pydantic schemas for FastAPI APIs.

These cover job submission, dataset management, telemetry, and PS <-> Worker contracts.
They are shared across both PS (server) and Worker clients.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl


# ----------------------------
# Job & training schemas
# ----------------------------
class TrainJobRequest(BaseModel):
    dataset_url: HttpUrl = Field(..., description="Remote dataset file (zip/csv/parquet)")
    algo: str = Field("ResNet50", description="Model architecture name")
    epochs: int = Field(10, ge=1)
    batch_size: int = Field(32, ge=1)
    learning_rate: float = Field(5e-4)
    optimizer: str = Field("adam")
    loss: str = Field("binary_crossentropy")
    validation_split: float = Field(0.2, ge=0.0, le=0.9)


class TrainJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # queued | running | completed | failed
    progress: float = Field(0.0, ge=0.0, le=1.0)
    metrics: Optional[Dict[str, Any]] = None


# ----------------------------
# Worker registration & heartbeats
# ----------------------------
class WorkerRegistration(BaseModel):
    worker_id: str
    host: str
    capabilities: Dict[str, Any] = Field(..., description="GPU, CPU cores, memory, etc.")


class WorkerHeartbeat(BaseModel):
    worker_id: str
    job_id: Optional[str] = None
    step: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None


# ----------------------------
# Parameter Server <-> Worker sync messages
# ----------------------------
class WeightUpdate(BaseModel):
    worker_id: str
    job_id: str
    step: int
    weights: List[float]  # flattened tensor for transmission


class AggregatedWeights(BaseModel):
    job_id: str
    step: int
    weights: List[float]


# ----------------------------
# Dataset schemas
# ----------------------------
class DatasetRegisterRequest(BaseModel):
    url: HttpUrl
    checksum: Optional[str] = None


class DatasetRegisterResponse(BaseModel):
    dataset_id: str
    status: str
    local_path: Optional[str] = None


# ----------------------------
# Telemetry
# ----------------------------
class TelemetryLog(BaseModel):
    event: str
    details: Dict[str, Any]
    timestamp: float
