"""
Telemetry Manager

Handles worker telemetry ingestion and aggregation.
- Store incoming metrics (loss, accuracy, etc.)
- Aggregate metrics for dashboards
- Expose query APIs for monitoring
"""

from sqlalchemy.orm import Session
from models.telemetry import Telemetry
from utils.logger import get_logger

logger = get_logger("telemetry-manager")


class TelemetryManager:
    def __init__(self, db: Session):
        self.db = db

    def record_metrics(self, job_id: int, worker_id: int, metrics: dict) -> Telemetry:
        telemetry = Telemetry(
            job_id=job_id,
            worker_id=worker_id,
            metrics=metrics,
        )
        self.db.add(telemetry)
        self.db.commit()
        self.db.refresh(telemetry)
        logger.info(f"Recorded telemetry for job {job_id} from worker {worker_id}")
        return telemetry

    def get_job_metrics(self, job_id: int):
        records = self.db.query(Telemetry).filter(Telemetry.job_id == job_id).all()
        return [r.metrics for r in records]

    def aggregate_job_metrics(self, job_id: int):
        records = self.db.query(Telemetry).filter(Telemetry.job_id == job_id).all()
        if not records:
            return {}
        # Example aggregation: average numeric values across workers
        aggregated = {}
        count = len(records)
        for record in records:
            for key, value in record.metrics.items():
                if isinstance(value, (int, float)):
                    aggregated[key] = aggregated.get(key, 0) + value
        for key in aggregated:
            aggregated[key] /= count
        logger.info(f"Aggregated metrics for job {job_id}: {aggregated}")
        return aggregated
