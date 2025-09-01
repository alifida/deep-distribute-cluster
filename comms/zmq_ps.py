import asyncio
import json
import logging
import zmq
import zmq.asyncio
import numpy as np
from typing import Dict, List, Optional
from copy import deepcopy
import time
from utils.db import SessionLocal
from models.db_models import TrainTrainingJob as Job
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("parameter-server")

# =======================
# Job Manager (unchanged)
# =======================
class JobManager:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}

    def create_job(self, job_id: str, dataset: List[dict]):
        if job_id in self.jobs:
            logger.warning(f"Job {job_id} already exists, replacing it")
        self.jobs[job_id] = {
            "batches": dataset.copy(),
            "workers": set(),
            "assigned_batches": {},
            "completed": False,
        }
        logger.info(f"Job {job_id} created with {len(dataset)} batches")

    def register_worker(self, job_id: str, worker_id: str):
        if job_id not in self.jobs:
            return False
        self.jobs[job_id]["workers"].add(worker_id)
        logger.info(f"Worker {worker_id} registered for job {job_id}")
        return True

    def assign_batch(self, job_id: str, worker_id: str) -> Optional[dict]:
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
        self.jobs: Dict[str, dict] = {}      # job_id -> job_state
        self.workers = {}
        self.training_started_at = None

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
                response = await self._handle_message(msg)
                await self.socket.send_multipart([identity, b"", json.dumps(response).encode()])
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
        elif msg_type == "get_available_jobs":
            return await self._get_available_jobs(msg)
        elif msg_type == "get_weights":
            return await self._get_weights(msg)
        elif msg_type == "report_metrics":
            return await self._report_metrics(msg)
        elif msg_type == "worker_done":
            return await self._worker_done(msg)
        else:
            return {"status": "error", "error": f"Unknown type {msg_type}"}

    # -----------------------
    # Job creation from Django
    # -----------------------
    async def _create_job(self, msg: dict) -> dict:
        job_data = msg.get("init_params", {}).get("job_data")
        dataset_details = msg.get("init_params", {}).get("dataset_details")
        if not job_data or not dataset_details:
            return {"status": "error", "error": "invalid_job_payload"}

        job_id = str(job_data["job_id"])
        params = job_data.get("parameter_settings", {}) or {}

        # Parse hyperparams with safe defaults
        def _to_int(x, d): 
            try: return int(x)
            except: return d
        def _to_float(x, d):
            try: return float(x)
            except: return d

        epochs = _to_int(params.get("epochs"), 1)
        batch_size = _to_int(params.get("batch_size"), 32)
        learning_rate = _to_float(params.get("learning_rate"), 1e-3)
        optimizer = (params.get("optimizer") or "adam").lower()
        loss_function = (params.get("loss_function") or "binary_crossentropy").lower()
        validation_split = max(0.0, min(1.0, _to_float(params.get("validation_split"), 0.2)))
        early_stopping_patience = _to_int(params.get("early_stopping_patience"), 10)
        dropout_rate = _to_float(params.get("dropout_rate"), 0.0)
        random_seed = _to_int(params.get("random_seed"), 42)

        # prefer explicit cluster setting; fallback to 1
        #try:
        #    num_partitions = int(params.get("cluster")) if params.get("cluster") else max(len(self.workers), 1)
        #except Exception:
        #    num_partitions = max(len(self.workers), 1)
        self.training_started_at = time.time()
        num_partitions = max(len(self.workers), 1)
        host_url = dataset_details.get("host_url", "").rstrip("/")

        # Build partitions: list of tuples (url,label) for 'train' and 'test'
        partitions = [{"train": [], "test": []} for _ in range(num_partitions)]

        def full_url(path: str) -> str:
            if not path:
                return path
            if path.startswith("http://") or path.startswith("https://"):
                return path
            return host_url + "/" + path.lstrip("/")

        # Round-robin assignment per class for balance
        for set_name in ("train", "test"):
            set_info = dataset_details.get(set_name, {})
            classwise = set_info.get("classwise_details", [])
            for class_idx, cls in enumerate(classwise):
                imgs = cls.get("preview_images", []) or []
                for i, rel_path in enumerate(imgs):
                    part_idx = i % num_partitions
                    partitions[part_idx][set_name].append((full_url(rel_path), class_idx))

        # Split each partition's TRAIN into train/val using validation_split
        rng = np.random.default_rng(random_seed)
        batches_per_partition = {}
        for p_idx, p in enumerate(partitions):
            batches_per_partition[p_idx] = {"train": [], "val": [], "test": []}

            # Shuffle training for randomness; keep test order
            train_examples = list(p["train"])
            rng.shuffle(train_examples)
            n_val = int(len(train_examples) * validation_split)
            val_examples = train_examples[:n_val]
            real_train = train_examples[n_val:]

            def to_batches(examples):
                out = []
                for i in range(0, len(examples), batch_size):
                    sl = examples[i:i + batch_size]
                    out.append({"X": [u for (u, _) in sl], "y": [int(l) for (_, l) in sl]})
                return out

            batches_per_partition[p_idx]["train"] = to_batches(real_train)
            batches_per_partition[p_idx]["val"] = to_batches(val_examples)
            batches_per_partition[p_idx]["test"] = to_batches(p["test"])

        # Build initial model and weights
        algo = params.get("algo", "tiny")
        input_shape = tuple(params.get("input_shape", (224, 224, 3)))

        model = self._build_model(algo, input_shape)
        model_json = model.to_json()
        weights_lists, weights_shapes = _serialize_weights(model.get_weights())

        # Build job state
        self.jobs[job_id] = {
            "num_partitions": num_partitions,
            "batches_per_partition": batches_per_partition,
            "partition_to_worker": {},
            "worker_to_partition": {},
            "batches": {},
            "epochs": epochs,
            "worker_epoch": {},
            "global_weights": None,
            "weights_version": 0,
            "status": "RUNNING",
            "hyperparams": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "optimizer": optimizer,
                "loss_function": loss_function,
                "validation_split": validation_split,
                "early_stopping_patience": early_stopping_patience,
                "dropout_rate": dropout_rate,
                "random_seed": random_seed,
                "image_size": tuple(dataset_details.get("train", {}).get("image_size", (224, 224))),
            },
            "global_weights_lists": weights_lists,
            "global_weights_shapes": weights_shapes,
            "agg_count": 0,
            "training_log": "",
            "training_log_history": [],
            "final_metrics": [],
            "done_workers": set(),
            "parameter_settings": params,
            "model_spec": {
                "algo": algo,
                "input_shape": input_shape,
                "model_json": model_json
            }
        }

        await self._update_job_status_db(job_id, "RUNNING")
        logger.info(f"Job {job_id} created: partitions={num_partitions}, epochs={epochs}")
        return {"status": "ok", "job_id": job_id, "partitions": num_partitions}

    # -----------------------
    # Worker registration
    # -----------------------
    async def _register_worker(self, msg: dict) -> dict:
        worker_id = msg.get("worker_id")
        job_id = msg.get("job_id")
        if not worker_id:
            return {"status": "error", "error": "missing_worker_id"}

        self.workers[worker_id] = {"registered": True}
        logger.info(f"Worker registered: {worker_id}")

        if not job_id:
            return {"status": "ok", "msg": "registered_no_job"}

        job = self.jobs.get(str(job_id))
        if not job:
            return {"status": "ok", "msg": "registered_job_not_found"}

        # Already assigned?
        if worker_id in job["worker_to_partition"]:
            p_idx = job["worker_to_partition"][worker_id]
            if worker_id not in job["batches"]:
                job["batches"][worker_id] = deepcopy(job["batches_per_partition"][p_idx])
            return {
                "status": "ok",
                "partition_index": int(p_idx),
                "num_train_batches": len(job["batches"][worker_id]["train"]),
                "num_val_batches": len(job["batches"][worker_id]["val"]),
                "num_test_batches": len(job["batches"][worker_id]["test"]),
                "hyperparams": job["hyperparams"],
                "model_json": job["model_spec"]["model_json"],
                "initial_weights": {
                    "lists": job["global_weights_lists"],
                    "shapes": job["global_weights_shapes"]
                }
            }

        # Find free partition
        all_parts = set(range(job["num_partitions"]))
        assigned_parts = set(job["partition_to_worker"].keys())
        free_parts = list(all_parts - assigned_parts)
        if not free_parts:
            return {"status": "ok", "msg": "all_partitions_assigned"}

        p_idx = min(free_parts)
        job["partition_to_worker"][p_idx] = worker_id
        job["worker_to_partition"][worker_id] = p_idx
        job["batches"][worker_id] = deepcopy(job["batches_per_partition"][p_idx])
        job["worker_epoch"][worker_id] = 1

        logger.info(f"Assigned partition {p_idx} of job {job_id} to worker {worker_id}")
        return {
            "status": "ok",
            "partition_index": int(p_idx),
            "num_train_batches": len(job["batches"][worker_id]["train"]),
            "num_val_batches": len(job["batches"][worker_id]["val"]),
            "num_test_batches": len(job["batches"][worker_id]["test"]),
            "hyperparams": job["hyperparams"],
            "model_json": job["model_spec"]["model_json"],
            "initial_weights": {
                "lists": job["global_weights_lists"],
                "shapes": job["global_weights_shapes"]
            }
        }

    # -----------------------
    # Batch serving with epoch cycling
    # -----------------------
    async def _get_batch(self, msg: dict) -> dict:
        worker_id = msg.get("worker_id")
        job_id = msg.get("job_id")
        set_name = msg.get("set", "train")
        if set_name not in ("train", "val", "test"):
            set_name = "train"

        if not worker_id or not job_id:
            return {"status": "error", "error": "missing_worker_or_job_id"}

        job = self.jobs.get(str(job_id))
        if not job:
            return {"status": "error", "error": "job_not_found"}

        wb = job["batches"].get(worker_id)
        if wb is None:
            return {"status": "error", "error": "no_batches_assigned_to_worker"}

        # still have a batch?
        if wb.get(set_name):
            batch = wb[set_name].pop(0)
            return {"status": "ok", **batch}

        # If training set exhausted, either finish epoch or finish job for the worker
        if set_name == "train":
            cur_epoch = job["worker_epoch"].get(worker_id, 1)
            if cur_epoch < job["epochs"]:
                # reset next epoch: reshuffle from template
                p_idx = job["worker_to_partition"][worker_id]
                new_epoch_train = deepcopy(job["batches_per_partition"][p_idx]["train"])
                # shuffle batch order each epoch
                rng = np.random.default_rng(job["hyperparams"]["random_seed"] + cur_epoch)
                rng.shuffle(new_epoch_train)
                wb["train"] = new_epoch_train
                job["worker_epoch"][worker_id] = cur_epoch + 1
                return {"status": "epoch_end", "epoch": cur_epoch}
            else:
                return {"status": "done", "epoch": cur_epoch}

        # If val/test exhausted
        return {"status": "empty", "message": f"No more {set_name} batches for worker"}

    # -----------------------
    # Weight exchange (per job)
    # -----------------------
    async def _update_weights(self, msg: dict) -> dict:
        job_id = str(msg.get("job_id"))
        worker_id = msg.get("worker_id")
        lists = msg.get("weights") or []
        shapes = [tuple(s) for s in (msg.get("shapes") or [])]

        job = self.jobs.get(job_id)
        if not job:
            return {"status": "error", "error": "job_not_found"}

        # running FedAvg
        gl_lists, gl_shapes = job["global_weights_lists"], job["global_weights_shapes"]
        agg_count = job["agg_count"]
        new_lists, new_shapes = _fedavg(gl_lists, gl_shapes, lists, shapes, agg_count)

        job["global_weights_lists"]  = new_lists
        job["global_weights_shapes"] = new_shapes
        job["agg_count"]             = agg_count + 1
        job["weights_version"]       = job.get("weights_version", -1) + 1

        logger.info(f"[Job {job_id}] aggregated weights from {worker_id}; version={job['weights_version']}, count={job['agg_count']}")
        return {"status": "ok", "version": job["weights_version"]}

    async def _get_weights(self, msg: dict) -> dict:
        job_id = str(msg.get("job_id"))
        job = self.jobs.get(job_id)
        if not job:
            return {"status": "error", "error": "job_not_found"}

        if not job["global_weights_lists"]:
            return {"status": "empty", "message": "no_global_weights_yet"}

        return {
            "status": "ok",
            "weights": job["global_weights_lists"],
            "shapes": job["global_weights_shapes"],
            "version": job["weights_version"]
        }
    # -----------------------
    # Metrics reporting & DB logging
    # -----------------------
    async def _report_metrics(self, msg: dict) -> dict:
        job_id = str(msg.get("job_id"))
        epoch  = int(msg.get("epoch", 0))
        scope  = msg.get("scope", "epoch")  # "epoch" or "final"
        logs   = msg.get("logs", {}) or {}
        support = int(msg.get("support", 0))  # optional sample count

        job = self.jobs.get(job_id)
        if not job:
            return {"status": "error", "error": "job_not_found"}

        entry = {"epoch": epoch, "logs": logs}
        if scope == "epoch":
            job["training_log_history"].append(entry)
            job["training_log"] = entry
        else:  # final
            job["final_metrics"].append({"metrics": logs, "support": support})

    # 🔥 Persist logs to DB
        await self._update_job_logs_db(
            job_id=job_id,
            latest_log=job.get("training_log"),
            history=job.get("training_log_history", [])
        )
        return {"status": "ok"}




    async def _worker_done(self, msg: dict) -> dict:
        job_id = str(msg.get("job_id"))
        worker_id = msg.get("worker_id")

        job = self.jobs.get(job_id)
        if not job:
            return {"status": "error", "error": "job_not_found"}

        job["done_workers"].add(worker_id)
        logger.info(f"[Job {job_id}] worker {worker_id} marked done ({len(job['done_workers'])}/{job['num_partitions']})")

        if len(job["done_workers"]) >= job["num_partitions"]:
            await self._finalize_job(job_id)

        return {"status": "ok"}

    def _build_model(self, algo: str, input_shape):
        # mirror worker's tiny model; allow dynamic HxW
        if not input_shape:
            input_shape = (None, None, 3)
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(16, 3, activation="relu")(inputs)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        return tf.keras.Model(inputs, outputs)

    async def _finalize_job(self, job_id: str):
        job = self.jobs.get(job_id)
        if not job:
            return

        model_spec = job.get("model_spec")
        if not model_spec:
            logger.error(f"[Job {job_id}] Missing model_spec, cannot finalize job.")
            return

        # Rebuild model
        model = self._build_model(
            model_spec.get("algo"),
            tuple(model_spec.get("input_shape", ())),
            **model_spec.get("extra_args", {})  # Pass extra params if stored
        )

        # Set global weights
        lists, shapes = job.get("global_weights_lists"), job.get("global_weights_shapes")
        if lists and shapes:
            arrays = _deserialize_weights(lists, shapes)
            if len(arrays) == len(model.get_weights()):
                model.set_weights(arrays)
            else:
                logger.error(f"[Job {job_id}] Weight mismatch, skipping evaluation")
                return

        # 🔥 Centralized evaluation
        all_test_batches = []
        for p in job.get("batches_per_partition", {}).values():
            all_test_batches.extend(p.get("test", []))

        logger.info(f"[Job {job_id}] Running centralized evaluation on {len(all_test_batches)} test batches...")
        results = _evaluate_model(model, all_test_batches)
        time_taken = int(time.time() - self.training_started_at) if self.training_started_at else 0
        results["time_taken"] = format_duration(time_taken)
        # Save model
        model_path = f"./{job_id}.h5"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: model.save(model_path))

        # Save results in DB
        await self._update_job_result_db(job_id, results)
        await self._update_job_status_db(job_id, "COMPLETED")
        logger.info(f"[Job {job_id}] finalized. Model saved at {model_path}. Results: {results}")


    async def _finalize_job__old(self, job_id: str):
        job = self.jobs.get(job_id)
        if not job:
            return

        # Ensure we rebuild the exact same model
        model_spec = job.get("model_spec")
        if not model_spec:
            logger.error(f"[Job {job_id}] Missing model_spec, cannot finalize job.")
            return

        # Rebuild model
        model = self._build_model(
            model_spec.get("algo"),
            tuple(model_spec.get("input_shape", ())),
            **model_spec.get("extra_args", {})  # Pass extra params if stored
        )

        # Check weight shapes before setting
        lists, shapes = job.get("global_weights_lists"), job.get("global_weights_shapes")
        if lists and shapes:
            arrays = _deserialize_weights(lists, shapes)
            expected = len(model.get_weights())
            received = len(arrays)
            if expected != received:
                logger.error(f"[Job {job_id}] Weight count mismatch: expected {expected}, got {received}. Skipping set_weights.")
            else:
                model.set_weights(arrays)

        model_path = f"./{job_id}.h5"

        loop = asyncio.get_event_loop()

        def save_task():
            model.save(model_path)
        await loop.run_in_executor(None, save_task)

        # Aggregate metrics as before
        finals = job.get("final_metrics") or []
        if finals:
            total = max(sum(m.get("support", 0) for m in finals), 1)
            keys = ["accuracy", "precision", "recall", "auc", "f1_score"]
            agg = {k: 0.0 for k in keys}
            for fm in finals:
                m = fm["metrics"]
                s = fm.get("support", 0) or 0
                w = (s / total) if s > 0 else (1.0 / len(finals))
                for k in keys:
                    if k in m and isinstance(m[k], (int, float)):
                        agg[k] += float(m[k]) * w
            results = {k: float(agg[k]) for k in keys}
        else:
            results = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "auc": 0.5, "f1_score": 0.0}

        await self._update_job_result_db(job_id, results)
        await self._update_job_status_db(job_id, "COMPLETED")

        logger.info(f"[Job {job_id}] finalized. Model saved at {model_path}. Results: {results}")


    async def _update_job_result_db(self, job_id: str, results: dict):
        loop = asyncio.get_event_loop()
        def db_task():
            db = SessionLocal()
            try:
                job = db.query(Job).filter(Job.id == job_id).first()
                if job:
                    job.result = json.dumps(results)
                    db.commit()
                    logger.info(f"Job {job_id} result saved to DB")
                else:
                    logger.warning(f"Job {job_id} not found for result update")
            finally:
                db.close()
        await loop.run_in_executor(None, db_task)


    # -----------------------
    # DB helpers
    # -----------------------
    async def _update_job_status_db(self, job_id: str, status: str):
        loop = asyncio.get_event_loop()
        def db_task():
            db = SessionLocal()
            try:
                job = db.query(Job).filter(Job.id == job_id).first()
                if job:
                    job.status = status
                    db.commit()
                else:
                    logger.warning(f"Job {job_id} not found in DB for status update")
            finally:
                db.close()
        await loop.run_in_executor(None, db_task)

    async def _update_job_logs_db(self, job_id: str, latest_log: dict, history: List[dict]):
        loop = asyncio.get_event_loop()
        latest_json = json.dumps(latest_log) if latest_log is not None else None
        history_json = json.dumps(history) if history is not None else "[]"

        def db_task():
            db = SessionLocal()
            try:
                job = db.query(Job).filter(Job.id == job_id).first()
                if job:
                    job.training_log = latest_json
                    job.training_log_history = history_json
                    db.commit()
                else:
                    logger.warning(f"Job {job_id} not found in DB for logs update")
            finally:
                db.close()
        await loop.run_in_executor(None, db_task)

    # -----------------------
    # Job discovery for workers
    # -----------------------
    async def _get_available_jobs(self, msg: dict) -> dict:
        worker_id = msg.get("worker_id")
        if not worker_id:
            return {"status": "error", "error": "missing_worker_id"}
        job_ids = [
            jid for jid, j in self.jobs.items()
            if j["status"] == "RUNNING" and worker_id not in j["worker_to_partition"]
        ]
        return {"status": "ok", "job_ids": job_ids}



def _evaluate_model(model, test_batches):
    """Run evaluation on PS using centralized test data."""
    y_true, y_pred = [], []

    for batch in test_batches:
        # Load X (images) and y
        X_batch = np.array([_load_image(x) for x in batch["X"]], dtype=np.float32)
        y_batch = np.array(batch["y"], dtype=np.int32)

        preds = model.predict(X_batch, verbose=0).ravel()
        y_true.extend(y_batch)
        y_pred.extend((preds > 0.5).astype(int))

    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "auc": 0.5, "f1_score": 0.0}

    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception:
        auc = 0.5

    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(auc),
        "f1_score": float(f1),
    }

def _load_image(url, target_size=(224, 224)):
    """Load and preprocess image for evaluation."""
    import requests
    from io import BytesIO
    from PIL import Image

    try:
        resp = requests.get(url, timeout=5)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img = img.resize(target_size)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr
    except Exception as e:
        logger.warning(f"Failed to load image {url}: {e}")
        return np.zeros((*target_size, 3), dtype=np.float32)


# NEW helpers (put below logger)
def _deserialize_weights(lists, shapes):
    return [np.array(w, dtype=np.float32).reshape(s) for w, s in zip(lists or [], shapes or [])]

def _serialize_weights(arrays):
    return [a.astype(np.float32).ravel().tolist() for a in arrays], [tuple(a.shape) for a in arrays]

def _fedavg(prev_lists, prev_shapes, new_lists, new_shapes, count):
    """Running FedAvg: new_global = (global*count + w) / (count+1)"""
    if not prev_lists:  # first arrival
        return new_lists, new_shapes
    prev = _deserialize_weights(prev_lists, prev_shapes)
    new  = _deserialize_weights(new_lists,  new_shapes)
    out  = [(p * count + n) / (count + 1.0) for p, n in zip(prev, new)]
    return _serialize_weights(out)

def format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds} sec"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} min"
    elif seconds < 86400:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours} hour{'s' if hours != 1 else ''}" + (f" {minutes} min" if minutes else "")
    else:
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        return f"{days} day{'s' if days != 1 else ''}" + (f" {hours} hour{'s' if hours != 1 else ''}" if hours else "")


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
