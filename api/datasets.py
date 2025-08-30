"""
API routes for dataset management.

Endpoints:
- POST /datasets -> register dataset
- GET /datasets -> list datasets
- GET /datasets/{dataset_id} -> get dataset details

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from utils.db import get_db
from models.db_models import Dataset
from schemas.schemas import DatasetCreate, DatasetOut

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/", response_model=DatasetOut)
def register_dataset(dataset_in: DatasetCreate, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_in.id).first()
    if dataset:
        raise HTTPException(status_code=400, detail="Dataset already registered")

    dataset = Dataset(
        id=dataset_in.id,
        name=dataset_in.name,
        path=dataset_in.path,
        format=dataset_in.format,
        description=dataset_in.description,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


@router.get("/", response_model=list[DatasetOut])
def list_datasets(db: Session = Depends(get_db)):
    return db.query(Dataset).all()


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset

"""