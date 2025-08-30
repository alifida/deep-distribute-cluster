"""
Orchestrator Service

Coordinates full job lifecycle:
1. Submits a new job (links dataset)
2. Assigns to worker(s)
3. Collects telemetry
4. Persists trained model

from sqlalchemy.orm import Session
from services.job_manager import JobManager
from services.dataset_manager import DatasetManager
from services.telemetry_manager import TelemetryManager
from services.model_manager import ModelManager
from utils.logger import get_logger

logger = get_logger("orchestrator")


class Orchestrator:
    def __init__(self, db: Session):
        self.db = db
        self.job_manager = JobManager(db)
        self.dataset_manager = DatasetManager(db)
        self.telemetry_manager = TelemetryManager(db)
        self.model_manager = ModelManager(db)

    def submit_job(self, dataset_id: int, model_name: str, version: str, description: str = None):
        # Ensure dataset exists
        dataset = self.dataset_manager.get_dataset(dataset_id)
        logger.info(f"Submitting job for dataset {dataset.id} ({dataset.name})")

        # Create job
        job = self.job_manager.create_job(dataset_id=dataset.id)

        # Assign job to worker(s)
        self.job_manager.assign_job(job.id, worker_id="auto")  # simplified for now

        return job

    def collect_telemetry(self, job_id: int):
        agg = self.telemetry_manager.aggregate_metrics(job_id)
        logger.info(f"Aggregated metrics for job {job_id}: {agg}")
        return agg

    def finalize_job(self, job_id: int, model_file: str, model_name: str, version: str, description: str = None):
        # Mark job completed
        self.job_manager.complete_job(job_id)

        # Register trained model
        model = self.model_manager.register_model(
            name=model_name,
            job_id=job_id,
            version=version,
            file_path=model_file,
            description=description,
        )
        logger.info(f"Job {job_id} finalized with model {model.id}")
        return model

"""