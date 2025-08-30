# Add or update this function in the file.
import json
from models.db_models import TrainingJob

def update_training_log(job_id, log_entry, training_log_history):
    job = db.query(TrainingJob).filter_by(id=job_id).first()
    job.training_log = json.dumps(log_entry)
    job.training_log_history = json.dumps(training_log_history)
    db.commit()