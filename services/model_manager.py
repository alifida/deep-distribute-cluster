"""
Model Manager

Handles model persistence and versioning.
- Register trained models
- Save model file
- Fetch model metadata
- Load model for inference
"""

from sqlalchemy.orm import Session
from models.model import Model
from utils.logger import get_logger
import os
import shutil

logger = get_logger("model-manager")


class ModelManager:
    def __init__(self, db: Session, storage_dir: str = "./models"):
        self.db = db
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def register_model(self, name: str, job_id: int, version: str, file_path: str, description: str = None) -> Model:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")

        # Copy to storage dir with unique name
        dest_path = os.path.join(self.storage_dir, f"{name}_v{version}.bin")
        shutil.copy(file_path, dest_path)

        model = Model(
            name=name,
            job_id=job_id,
            version=version,
            path=dest_path,
            description=description,
        )
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        logger.info(f"Registered model {model.id} ({name} v{version})")
        return model

    def get_model(self, model_id: int) -> Model:
        model = self.db.query(Model).filter(Model.id == model_id).first()
        if not model:
            raise ValueError("Invalid model ID")
        return model

    def load_model(self, model_id: int):
        model = self.get_model(model_id)
        if not os.path.exists(model.path):
            raise FileNotFoundError(f"Model file missing: {model.path}")
        logger.info(f"Loading model {model.id} from {model.path}")
        # Stub: Replace with ML framework specific load logic
        return open(model.path, "rb").read()