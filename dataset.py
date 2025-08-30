"""
Dataset management module.

Provides endpoints for:
- dataset registration (by admin or PS)
- dataset retrieval (by workers)
- caching dataset metadata
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import uuid
import time

from api.schemas import DatasetRegistration

router = APIRouter()

# In-memory dataset registry (to be backed by DB/cache later)
DATASETS: Dict[str, Dict[str, Any]] = {}


# ----------------------------
# Dataset registration
# ----------------------------
@router.post("/register")
def register_dataset(req: DatasetRegistration):
    dataset_id = str(uuid.uuid4())
    DATASETS[dataset_id] = {
        "name": req.name,
        "path": req.path,
        "description": req.description,
        "format": req.format,
        "created": time.time(),
    }
    return {"status": "registered", "dataset_id": dataset_id}


# ----------------------------
# Dataset listing & retrieval
# ----------------------------
@router.get("/list")
def list_datasets():
    return [{"id": ds_id, **meta} for ds_id, meta in DATASETS.items()]


@router.get("/get/{dataset_id}")
def get_dataset(dataset_id: str):
    dataset = DATASETS.get(dataset_id)
    if not dataset:
        raise HTTPException(404, detail="Dataset not found")
    return {"dataset_id": dataset_id, **dataset}
