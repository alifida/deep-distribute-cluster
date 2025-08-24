import asyncio
import json
import logging
import zmq
import zmq.asyncio
import numpy as np
from typing import Dict, List, Optional

from copy import deepcopy


from utils.db import SessionLocal
from models.db_models import TrainTrainingJob as Job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("parameter-server")


# =======================
# Job Manager
# =======================
class JobManager:
    """Tracks jobs, workers, and batch assignment."""

    def __init__(self):
        self.jobs: Dict[str, dict] = {}  # job_id -> job_info

    def create_job(self, job_id: str, dataset: List[dict]):
        """Create a new job with dataset (list of {X:..., y:...})."""
        if job_id in self.jobs:
            logger.warning(f"Job {job_id} already exists, replacing it")
        self.jobs[job_id] = {
            "batches": dataset.copy(),
            "workers": set(),
            "assigned_batches": {},  # worker_id -> list of assigned batch idx
            "completed": False,
        }
        logger.info(f"Job {job_id} created with {len(dataset)} batches")

    def register_worker(self, job_id: str, worker_id: str):
        """Register worker for a job."""
        if job_id not in self.jobs:
            return False
        self.jobs[job_id]["workers"].add(worker_id)
        logger.info(f"Worker {worker_id} registered for job {job_id}")
        return True

    def assign_batch(self, job_id: str, worker_id: str) -> Optional[dict]:
        """Return next batch for a worker, or None if no more."""
        if job_id not in self.jobs:
            return None
        job = self.jobs[job_id]
        if not job["batches"]:
            return None
        batch = job["batches"].pop(0)
        job["assigned_batches"].setdefault(worker_id, []).append(batch)
        logger.info(f"Assigned batch to worker {worker_id} for job {job_id}")
        return batch

    def all_batches_done(self, job_id: str) -> bool:
        """Check if all batches are completed."""
        return job_id in self.jobs and not self.jobs[job_id]["batches"]


# =======================
# Parameter Server
# =======================
class ParameterServerZMQ:
    def __init__(self, bind_addr: str = "tcp://0.0.0.0:5555"):
        self.bind_addr = bind_addr
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.running = False

        self.job_manager = JobManager()
        self.jobs = {}  
        self.workers = {}
        self.global_weights = None
        self.weights_version = -1

    async def start(self):
        self.socket.bind(self.bind_addr)
        self.running = True
        logger.info(f"Parameter Server running at {self.bind_addr}")
        asyncio.create_task(self._listen())

    async def stop(self):
        self.running = False
        self.socket.close()
        await self.context.term()
        logger.info("Parameter Server stopped")

    async def _listen(self):
        while self.running:
            try:
                frames = await self.socket.recv_multipart()
                if len(frames) < 3:
                    logger.error(f"Invalid message framing: {frames}")
                    continue
                identity, _, payload = frames
                msg = json.loads(payload.decode())
                logger.info(f"Received: {msg}")

                response = await self._handle_message(msg)
                await self.socket.send_multipart(
                    [identity, b"", json.dumps(response).encode()]
                )
            except Exception as e:
                logger.exception(f"Error processing message: {e}")

    async def _handle_message(self, msg: dict) -> dict:
        msg_type = msg.get("type")
        if msg_type == "register":
            return await self._register_worker(msg)
        elif msg_type == "get_batch":
            return await self._get_batch(msg)
        elif msg_type == "send_weights":
            return await self._update_weights(msg)
        elif msg_type == "create_job":
            return await self._create_job(msg)
        elif msg_type == "get_available_jobs":   # <-- NEW
            return await self._get_available_jobs(msg)
        else:
            return {"status": "error", "error": f"Unknown type {msg_type}"}
        
    async def _create_job(self, msg: dict) -> dict:
        """
        Create a job from the Django payload.
        Distribute images per set and per class equally across partitions (cluster size).
        """
        job_data = msg.get("init_params", {}).get("job_data")
        dataset_details = msg.get("init_params", {}).get("dataset_details")
        if not job_data or not dataset_details:
            return {"status": "error", "error": "invalid_job_payload"}

        job_id = str(job_data["job_id"])
        params = job_data.get("parameter_settings", {}) or {}
        # prefer explicit cluster setting; fallback to 1
        try:
            num_partitions = int(params.get("cluster")) if params.get("cluster") else max(len(self.workers), 1)
        except Exception:
            num_partitions = max(len(self.workers), 1)

        batch_size = int(params.get("batch_size", 32))

        host_url = dataset_details.get("host_url", "").rstrip("/")  # prefix for preview_images

        # Prepare partitions: each partition has structure { 'train': [], 'test': [] }
        partitions = [ {"train": [], "test": []} for _ in range(num_partitions) ]

        # Helper to build full URL
        def full_url(path: str) -> str:
            if not path:
                return path
            if path.startswith("http://") or path.startswith("https://"):
                return path
            return host_url + "/" + path.lstrip("/")

        # For each set and each class, round-robin assign image URLs to partitions
        for set_name in ("train", "test"):
            set_info = dataset_details.get(set_name, {})
            classwise = set_info.get("classwise_details", [])
            for class_idx, cls in enumerate(classwise):
                imgs = cls.get("preview_images", []) or []
                for i, rel_path in enumerate(imgs):
                    part_idx = i % num_partitions
                    partitions[part_idx][set_name].append((full_url(rel_path), class_idx))

        # Now create batches for each partition and each set
        batches_per_partition = {}
        for p_idx, p in enumerate(partitions):
            batches_per_partition[p_idx] = {"train": [], "test": []}
            for set_name in ("train", "test"):
                examples = p[set_name]  # list of (url, label)
                # For training, shuffle to randomize; for test keep order
                if set_name == "train":
                    # RNG can be seeded with job_id for reproducibility if desired
                    np.random.shuffle(examples)
                # convert list of tuples into batches (X:list_of_urls, y:list_of_labels)
                for i in range(0, len(examples), batch_size):
                    slice_ = examples[i:i+batch_size]
                    X_urls = [t[0] for t in slice_]
                    y_labels = [t[1] for t in slice_]
                    batches_per_partition[p_idx][set_name].append({"X": X_urls, "y": y_labels})

        # Build job state
        self.jobs[job_id] = {
            "num_partitions": num_partitions,
            "batches_per_partition": batches_per_partition,   # immutable partition batches
            "partition_to_worker": {},   # partition_idx -> worker_id
            "worker_to_partition": {},   # worker_id -> partition_idx
            "batches": {},               # worker_id -> {'train': [...], 'test': [...]}
            "global_weights": None,
            "weights_version": -1,
            "training_log": "",
            "training_log_history": [],
            "status": "RUNNING",
            "parameter_settings": params
        }

        # persist job status in DB
        await self._update_job_status_db(job_id, "RUNNING")
        logger.info(f"Job {job_id} created: partitions={num_partitions}, total_batches={sum(len(batches_per_partition[i]['train']) for i in batches_per_partition)}")
        return {"status": "ok", "job_id": job_id, "partitions": num_partitions}

    async def _register_worker(self, msg: dict) -> dict:
        """
        Register worker. If job_id present, assign an unassigned partition for that job.
        Response includes assigned partition index and number of batches (train/test).
        """
        worker_id = msg.get("worker_id")
        job_id = msg.get("job_id")  # optional - worker can register for a job
        if not worker_id:
            return {"status": "error", "error": "missing_worker_id"}

        # always record worker in global registry
        self.workers[worker_id] = {"registered": True}
        logger.info(f"Worker registered: {worker_id}")

        if not job_id:
            # no job requested, just acknowledge registration
            return {"status": "ok", "msg": "registered_no_job"}

        job = self.jobs.get(str(job_id))
        if not job:
            # job not found yet; worker is registered globally, but no partition assigned
            return {"status": "ok", "msg": "registered_job_not_found"}

        # If worker already mapped, return existing assignment
        if worker_id in job["worker_to_partition"]:
            p_idx = job["worker_to_partition"][worker_id]
            assigned_batches = deepcopy(job["batches_per_partition"][p_idx])
            # save assigned batch lists under worker's id if not already saved
            if worker_id not in job["batches"]:
                job["batches"][worker_id] = assigned_batches
            return {
                "status": "ok",
                "partition_index": int(p_idx),
                "num_train_batches": len(assigned_batches["train"]),
                "num_test_batches": len(assigned_batches["test"])
            }

        # find an unassigned partition index
        all_parts = set(range(job["num_partitions"]))
        assigned_parts = set(job["partition_to_worker"].keys())
        free_parts = list(all_parts - assigned_parts)

        if not free_parts:
            # all partitions already assigned — worker registers but gets no data
            return {"status": "ok", "msg": "all_partitions_assigned"}

        p_idx = min(free_parts)  # pick smallest free index for determinism
        job["partition_to_worker"][p_idx] = worker_id
        job["worker_to_partition"][worker_id] = p_idx

        # copy the prepared batches for this partition into worker-specific store
        job["batches"][worker_id] = deepcopy(job["batches_per_partition"][p_idx])

        logger.info(f"Assigned partition {p_idx} of job {job_id} to worker {worker_id}")
        return {
            "status": "ok",
            "partition_index": int(p_idx),
            "num_train_batches": len(job["batches"][worker_id]["train"]),
            "num_test_batches": len(job["batches"][worker_id]["test"])
        }

    async def _get_batch(self, msg: dict) -> dict:
        """
        Worker requests next batch.
        msg must include 'job_id' and 'worker_id'; optional 'set' param (train/test)
        """
        worker_id = msg.get("worker_id")
        job_id = msg.get("job_id")
        set_name = msg.get("set", "train")
        if set_name not in ("train", "test"):
            set_name = "train"

        if not worker_id or not job_id:
            return {"status": "error", "error": "missing_worker_or_job_id"}

        job = self.jobs.get(str(job_id))
        if not job:
            return {"status": "error", "error": "job_not_found"}

        worker_batches = job["batches"].get(worker_id)
        if not worker_batches:
            return {"status": "error", "error": "no_batches_assigned_to_worker"}

        if worker_batches.get(set_name):
            batch = worker_batches[set_name].pop(0)
            return {"status": "ok", **batch}
        else:
            return {"status": "empty", "message": f"No more {set_name} batches for worker"}
        

    async def _update_weights(self, msg: dict) -> dict:
        self.global_weights = msg.get("weights")
        self.weights_version += 1
        logger.info(f"Updated weights to version {self.weights_version}")
        return {"status": "ok"}

    async def _update_job_status_db(self, job_id: str, status: str):
        """
        Update a job's status in the database asynchronously.
        Runs DB operations in a thread to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()

        def db_task():
            db = SessionLocal()
            try:
                job = db.query(Job).filter(Job.id == job_id).first()
                if job:
                    job.status = status
                    db.commit()
                    logger.info(f"Job {job_id} status updated to {status} in DB")
                else:
                    logger.warning(f"Job {job_id} not found in DB for status update")
            finally:
                db.close()

        await loop.run_in_executor(None, db_task)

    async def _get_available_jobs(self, msg: dict) -> dict:
        """
        Worker asks for available jobs that are RUNNING and not yet assigned to it.
        Returns a list of job_ids.
        """
        worker_id = msg.get("worker_id")
        if not worker_id:
            return {"status": "error", "error": "missing_worker_id"}

        available_jobs = []
        for job_id, job in self.jobs.items():
            # Check if job is running and worker has no partition assigned
            if job["status"] == "RUNNING" and worker_id not in job["worker_to_partition"]:
                available_jobs.append(job_id)

        return {"status": "ok", "job_ids": available_jobs}


# =======================
# Entrypoint
# =======================
async def run_server():
    ps = ParameterServerZMQ()
    await ps.start()
    try:
        while True:
            await asyncio.sleep(10)
    except KeyboardInterrupt:
        await ps.stop()


if __name__ == "__main__":
    asyncio.run(run_server())
