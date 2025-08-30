"""
Dataset Manager

Handles dataset registration and loading for jobs.
- Register dataset metadata (path, type, description)
- Fetch dataset by ID
- Load data for training (stub for now)
"""

from sqlalchemy.orm import Session
from models.dataset import Dataset
from utils.logger import get_logger
import os

logger = get_logger("dataset-manager")


class DatasetManager:
    def __init__(self, db: Session):
        self.db = db

    def register_dataset(self, name: str, path: str, dtype: str = "csv", description: str = None) -> Dataset:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        dataset = Dataset(name=name, path=path, type=dtype, description=description)
        self.db.add(dataset)
        self.db.commit()
        self.db.refresh(dataset)
        logger.info(f"Registered dataset {dataset.id} ({name})")
        return dataset

    def get_dataset(self, dataset_id: int) -> Dataset:
        dataset = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise ValueError("Invalid dataset ID")
        return dataset

    def load_dataset(self, dataset_id: int):
        dataset = self.get_dataset(dataset_id)
        logger.info(f"Loading dataset {dataset.id} from {dataset.path}")
        # Stub loader: actual implementation depends on format
        if dataset.type == "csv":
            import pandas as pd
            return pd.read_csv(dataset.path)
        elif dataset.type == "json":
            import json
            with open(dataset.path) as f:
                return json.load(f)
        else:
            raise NotImplementedError(f"Dataset type {dataset.type} not supported")
