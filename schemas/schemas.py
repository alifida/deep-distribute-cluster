"""
Pydantic schemas for API requests and responses.

Defines:
- Job schemas
- Worker schemas
- Dataset schemas
- Telemetry schemas
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any, List


# ------------------- Job -------------------
class JobCreate(BaseModel):
    model: str
    dataset: str


class JobUpdate(BaseModel):
    status: Optional[str]
    progress: Optional[float]


class JobOut(BaseModel):
    id: str
    model: str
    dataset: str
    status: str
    progress: float

    class Config:
        orm_mode = True


# ------------------- Worker -------------------
class WorkerRegister(BaseModel):
    id: str


class WorkerOut(BaseModel):
    id: str
    status: str
    last_heartbeat: Optional[float]

    class Config:
        orm_mode = True


# ------------------- Dataset -------------------
class DatasetCreate(BaseModel):
    id: str
    name: str
    path: str
    format: str
    description: Optional[str]


class DatasetOut(BaseModel):
    id: str
    name: str
    path: str
    format: str
    description: Optional[str]

    class Config:
        orm_mode = True


# ------------------- Telemetry -------------------
class TelemetryLogCreate(BaseModel):
    worker_id: str
    job_id: str
    timestamp: float
    level: str = "INFO"
    message: Optional[str]
    metrics: Optional[Dict[str, Any]]


class TelemetryLogOut(BaseModel):
    id: int
    worker_id: str
    job_id: str
    timestamp: float
    level: str
    message: Optional[str]
    metrics: Optional[Dict[str, Any]]

    class Config:
        orm_mode = True
