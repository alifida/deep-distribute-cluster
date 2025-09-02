# zmq_worker.py
"""
Async ZMQ Worker with disk-backed prefetch cache.
Compatible with the zmq_ps.py you provided.

Features:
- Asynchronous prefetch (asyncio.Task) of future train batches into a disk cache.
- Reuse cached image files across epochs (no re-download).
- Resize images to model's expected input shape before training.
- Sends weights to PS after each training step (as in your previous code).
- Detailed logging at each stop so you can see progress and detect stalls.
"""
from utils.config import settings
from utils.db import SessionLocal
import asyncio
import json
import logging
import uuid
import os
import hashlib
from typing import List, Tuple, Optional, Any, Dict

import aiohttp
import numpy as np
import zmq.asyncio
from PIL import Image
import tensorflow as tf
from services.keras_model_service import KerasCatalogService
from models.db_models import TrainClusterNode as ClusterNode



logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("worker")


# -------------------------
# Helpers: optimizer, loss
# -------------------------
def make_optimizer(name: str, lr: float):
    name = (name or "adam").lower()
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    if name in ("adam", "adamw"):
        return tf.keras.optimizers.Adam(learning_rate=lr)
    return tf.keras.optimizers.Adam(learning_rate=lr)


def make_loss(name: str):
    name = (name or "binary_crossentropy").lower()
    if name in ("binary_crossentropy", "bce"):
        return tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return tf.keras.losses.BinaryCrossentropy(from_logits=False)


# -------------------------
# Disk-backed image cache
# -------------------------
class DataCache:
    """
    Caches remote image URLs to disk. Only file paths are kept in memory.
    Filenames use md5(url) to dedupe.
    """

    def __init__(self, job_id: str, cache_root: Optional[str] = None):
        self.job_id = str(job_id)
        root = cache_root or os.environ.get("WORKER_CACHE_DIR", "/tmp/zmq_cache")
        self.base_dir = os.path.join(root, f"job_{self._short_hash(self.job_id)}")
        os.makedirs(self.base_dir, exist_ok=True)
        self._loop = asyncio.get_event_loop()

    @staticmethod
    def _short_hash(s: str) -> str:
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

    def _path_for_url(self, url: str) -> str:
        h = hashlib.md5(url.encode("utf-8")).hexdigest()
        name = os.path.basename(url.split("?")[0]) or h
        _, ext = os.path.splitext(name)
        ext = ext if ext.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff") else ".jpg"
        return os.path.join(self.base_dir, f"{h}{ext}")

    async def _write_bytes(self, path: str, data: bytes):
        def _sync_write(p, b):
            with open(p, "wb") as f:
                f.write(b)
        await self._loop.run_in_executor(None, _sync_write, path, data)

    async def _download_one(self, session: aiohttp.ClientSession, url: str, dest: str) -> str:
        # skip if already present with non-zero size
        try:
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                logger.debug("Cache hit for %s", url)
                return dest
        except Exception:
            pass

        try:
            async with session.get(url, timeout=30) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"bad status {resp.status} for {url}")
                data = await resp.read()
            await self._write_bytes(dest, data)
            logger.debug("Downloaded -> %s", dest)
            return dest
        except Exception as e:
            logger.warning("Failed to download %s: %s", url, e)
            # write placeholder so we don't keep retrying aggressively
            try:
                await self._write_bytes(dest, b"")
            except Exception:
                pass
            return dest

    async def download_urls(self, urls: List[str], max_concurrency: int = 12) -> List[str]:
        """Download list of URLs concurrently (bounded) and return corresponding local paths (same order)."""
        if not urls:
            return []
        sem = asyncio.Semaphore(max_concurrency)
        async with aiohttp.ClientSession() as session:
            async def _dl(u):
                dest = self._path_for_url(u)
                async with sem:
                    return await self._download_one(session, u, dest)
            tasks = [asyncio.create_task(_dl(u)) for u in urls]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            return results

    def load_batch_from_paths(self, paths: List[str], target_size: Tuple[int, int]) -> np.ndarray:
        """Load local files, resize to target_size (HxW), normalize to [0,1], return numpy array."""
        if not paths:
            return np.zeros((0, target_size[0], target_size[1], 3), dtype=np.float32)
        out = []
        h, w = target_size
        for p in paths:
            try:
                if os.path.getsize(p) > 0:
                    img = Image.open(p).convert("RGB")
                else:
                    img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
            except Exception:
                img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
            img = img.resize((w, h), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            out.append(arr)
        return np.stack(out, axis=0)


# -------------------------
# Prefetcher (async task)
# -------------------------
class TrainPrefetcher:
    """
    Async prefetcher: fetches training batches from PS and caches them,
    puts {"paths": [...], "y": [...]} onto an asyncio.Queue.
    Also pushes control messages: {"control": "epoch_end", "epoch": N} or {"control": "done", "epoch": N}
    """

    def __init__(self, worker_client: "WorkerZMQ", job_id: str, cache: DataCache, queue: asyncio.Queue, max_download_concurrency: int = 12):
        self.worker_client = worker_client
        self.job_id = str(job_id)
        self.cache = cache
        self.queue = queue
        self.max_download_concurrency = max_download_concurrency
        self._task: Optional[asyncio.Task] = None
        self._stopped = False

    def start(self):
        if self._task is None or self._task.done():
            self._stopped = False
            self._task = asyncio.create_task(self._run())
            logger.info("Prefetcher started")

    async def stop(self):
        self._stopped = True
        if self._task:
            try:
                await self._task
            except Exception:
                logger.debug("Prefetcher stop encountered exception")

    async def _run(self):
        try:
            while not self._stopped:
                batch = await self.worker_client.fetch_batch(self.job_id, "train")
                st = batch.get("status")
                if st == "ok":
                    urls = batch.get("X", []) or []
                    y = batch.get("y", []) or []
                    # download to disk, returns local paths
                    paths = await self.cache.download_urls(urls, max_concurrency=self.max_download_concurrency)
                    await self.queue.put({"paths": paths, "y": y})
                    logger.debug("Prefetcher queued batch (len=%d)", len(paths))
                    continue
                if st == "epoch_end":
                    await self.queue.put({"control": "epoch_end", "epoch": int(batch.get("epoch", 0))})
                    logger.info("Prefetcher got epoch_end (epoch=%s)", batch.get("epoch"))
                    break
                if st == "done":
                    await self.queue.put({"control": "done", "epoch": int(batch.get("epoch", 0))})
                    logger.info("Prefetcher got done (epoch=%s)", batch.get("epoch"))
                    break
                if st == "empty":
                    # small backoff
                    await asyncio.sleep(0.3)
                    continue
                # unknown
                await asyncio.sleep(0.3)
        except asyncio.CancelledError:
            logger.debug("Prefetcher cancelled")
        except Exception:
            logger.exception("Prefetcher error")


# -------------------------
# Worker (ZMQ)
# -------------------------
class WorkerZMQ:
    def __init__(self, server_addr: str = settings.PARAMETER_SERVER_URL, worker_id: Optional[str] = None):
        self.server_addr = server_addr
        worker_node = get_worker(3)  # cluster_id hardcoded for now
        if not worker_node :
            logger.exception("unable to start worker;  not registered in DB")
            return
        #self.worker_id = worker_id or str(uuid.uuid4())
        self.worker_id = worker_node.node_type +"_"+ str(worker_node.id)
        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.DEALER)
        # important: set identity so PS ROUTER receives identity frame
        self.socket.setsockopt(zmq.IDENTITY, self.worker_id.encode())
        self.socket.connect(self.server_addr)
        self.running = False

        # ML objects set after register
        self.model: Optional[tf.keras.Model] = None
        self.optimizer = None
        self.loss_fn = None
        self.hyperparams: Dict[str, Any] = {}
        self.target_size: Tuple[int, int] = (224, 224)

    async def _send_and_recv(self, message: dict, timeout: float = 20.0) -> dict:
        """Send with correct framing expected by your PS (empty delimiter) and receive reply."""
        try:
            await self.socket.send_multipart([b"", json.dumps(message).encode()])
            parts = await asyncio.wait_for(self.socket.recv_multipart(), timeout=timeout)
            # parts: [identity, b'', payload]
            reply = parts[-1]
            return json.loads(reply.decode())
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for server response")
            return {"status": "error", "error": "timeout"}
        except Exception as e:
            logger.exception("Error send/recv: %s", e)
            return {"status": "error", "error": str(e)}

    # RPC wrappers
    async def register(self, job_id: Optional[str] = None) -> dict:

        return await self._send_and_recv({"type": "register", "worker_id": self.worker_id, "job_id": job_id})

    async def get_available_jobs(self) -> dict:
        return await self._send_and_recv({"type": "get_available_jobs", "worker_id": self.worker_id})

    async def fetch_batch(self, job_id: str, set_name: str = "train") -> dict:
        return await self._send_and_recv({"type": "get_batch", "worker_id": self.worker_id, "job_id": job_id, "set": set_name})

    async def send_weights(self, job_id: str, lists: List[List[float]], shapes: List[Tuple[int, ...]]) -> dict:
        return await self._send_and_recv({"type": "send_weights", "job_id": job_id, "worker_id": self.worker_id, "weights": lists, "shapes": shapes})

    async def get_weights(self, job_id: str) -> dict:
        return await self._send_and_recv({"type": "get_weights", "job_id": job_id, "worker_id": self.worker_id})

    async def report_metrics(self, job_id: str, epoch: int, logs: dict, scope: str = "epoch", support: int = 0) -> dict:
        msg = {"type": "report_metrics", "job_id": job_id, "worker_id": self.worker_id, "epoch": epoch, "scope": scope, "logs": logs, "support": support}
        return await self._send_and_recv(msg)

    async def worker_done(self, job_id: str) -> dict:
        return await self._send_and_recv({"type": "worker_done", "job_id": job_id, "worker_id": self.worker_id})

    # -------------------------
    # Hyperparams / model utility
    # -------------------------
    def _apply_hyperparams(self, hp: dict):
        seed = int(hp.get("random_seed", 42))
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.hyperparams = hp
        # prepare optimizer & loss objects (lr will be used to instantiate optimizer)
        self.optimizer = make_optimizer(hp.get("optimizer"), float(hp.get("learning_rate", 1e-3)))
        self.loss_fn = make_loss(hp.get("loss_function"))

    async def _train_on_cached_batch(self, job_id: str, cache: DataCache, paths: List[str], y_labels: List[int]) -> float:
        X = cache.load_batch_from_paths(paths, target_size=self.target_size)
        if X.shape[0] == 0:
            return 0.0
        y = np.array(y_labels, dtype=np.float32)

        X_tensor = tf.convert_to_tensor(X)
        y_tensor = tf.convert_to_tensor(y)

        with tf.GradientTape() as tape:
            preds = tf.reshape(self.model(X_tensor, training=True), [-1])
            loss = self.loss_fn(y_tensor, preds)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # push weights to PS immediately
        weights = self.model.get_weights()
        lists = [w.flatten().astype(float).tolist() for w in weights]
        shapes = [tuple(w.shape) for w in weights]
        await self.send_weights(job_id, lists, shapes)
        return float(loss.numpy()) if hasattr(loss, "numpy") else float(loss)

    async def _eval_split(self, job_id: str, set_name: str, cache: DataCache):


        """Worker-side evaluation is disabled because PS handles evaluation centrally."""
        logger.info(f"Skipping evaluation for split '{split}' on worker side.")
        return 0.0, 0.0


        """Evaluate across all batches in given split. Uses cache for val/test as well."""
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score  # local import
        y_true_all, y_pred_all = [], []
        loss_sum, n = 0.0, 0
        while True:
            batch = await self.fetch_batch(job_id, set_name)
            st = batch.get("status")
            if st != "ok":
                break
            urls = batch.get("X", []) or []
            paths = await cache.download_urls(urls, max_concurrency=8)
            X = cache.load_batch_from_paths(paths, target_size=self.target_size)
            y = np.array(batch.get("y", []), dtype=np.float32)
            preds = tf.reshape(self.model(tf.convert_to_tensor(X), training=False), [-1]).numpy()
            try:
                loss_val = float(self.loss_fn(tf.convert_to_tensor(y), tf.convert_to_tensor(preds)).numpy().item())
            except Exception:
                loss_val = 0.0
            loss_sum += loss_val * len(y)
            y_true_all.extend(y.tolist())
            y_pred_all.extend((preds > 0.5).astype(int).tolist())
            n += len(y)
        if n == 0:
            return 0.0, 0.0
        acc = float(np.mean(np.array(y_true_all) == np.array(y_pred_all)))
        avg_loss = (loss_sum / n)
        # we won't compute precision/recall here unless requested by PS aggregation
        return avg_loss, acc



    def _build_model(self, algo: str, input_shape):
        if not input_shape:
            input_shape = (150, 150, 3)  # Default shape if none is provided

        try:
            # Try to load from predefined Keras applications
            base_model = KerasCatalogService.get_model_object(algo, input_shape=input_shape)
            return base_model

        except Exception as e:
            logger.error(f"Error loading model '{algo}': {e}. Falling back to tiny custom CNN.")
            # Fallback to tiny custom CNN if algo is not known
            inputs = tf.keras.Input(shape=input_shape)
            x = tf.keras.layers.Conv2D(16, 3, activation="relu")(inputs)
            x = tf.keras.layers.MaxPool2D()(x)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            return tf.keras.Model(inputs, outputs)


    # -------------------------
    # Main loop
    # -------------------------
    async def run(self):
        self.running = True
        logger.info("Worker %s initialized", self.worker_id)

        try:
            # 1) Robust job-acquire loop: keep polling and attempt to register to avoid race
            job_id = None
            reg: Optional[Dict[str, Any]] = None
            while self.running and reg is None:
                avail = await self.get_available_jobs()
                if avail.get("status") == "ok":
                    job_ids = avail.get("job_ids", []) or []
                    if job_ids:
                        # Try to register any available job (handles races when multiple workers compete)
                        for candidate in job_ids:
                            reg_try = await self.register(candidate)
                            if reg_try.get("status") == "ok":
                                reg = reg_try
                                job_id = candidate
                                break
                        if reg is None:
                            logger.info("Jobs listed but registration failed (likely race). Retrying...")
                    else:
                        logger.info("No jobs available, retrying...")
                else:
                    logger.warning("Failed to get available jobs: %s", avail)
                await asyncio.sleep(2.0)

            if reg is None:
                logger.info("Exiting worker run without a job (shutdown or no assignment).")
                return

            logger.info("Registered worker: %s", reg.get("msg", "ok"))

            hp = reg.get("hyperparams", {}) or {}
            model_spec = reg.get("model_spec", {}) or {}
            #model_json = reg.get("model_json")
            init_w = reg.get("initial_weights", {}) or {}

            # apply hyperparams
            self._apply_hyperparams(hp)

            # Rebuild model
            try:
                #self.model = tf.keras.models.model_from_json(model_json)
                self.model = KerasCatalogService.build_model(model_spec.get("algo", "tiny"), tuple(model_spec.get("input_shape", (150, 150, 3))))
            except Exception:
                logger.exception("Failed to build model ")
                return

            # Determine target input size from model's input shape if available
            try:
                inp_shape = self.model.input_shape  # (None, H, W, C) or (None, None, None, C)
                if inp_shape and len(inp_shape) >= 3 and inp_shape[1] and inp_shape[2]:
                    self.target_size = (int(inp_shape[1]), int(inp_shape[2]))
                else:
                    # fallback to hyperparams image_size if provided (h,w)
                    ps_img = hp.get("image_size") or hp.get("image_size", (224, 224))
                    if isinstance(ps_img, (list, tuple)) and len(ps_img) >= 2:
                        self.target_size = (int(ps_img[0]), int(ps_img[1]))
                    else:
                        self.target_size = (224, 224)
            except Exception:
                self.target_size = (224, 224)
            logger.info("Using target image size (H,W) = %s", self.target_size)

            # If initial weights provided by PS, set them
            if init_w.get("lists") and init_w.get("shapes"):
                try:
                    arrays = [np.array(l, dtype=np.float32).reshape(tuple(s)) for l, s in zip(init_w["lists"], init_w["shapes"])]
                    if len(arrays) == len(self.model.get_weights()):
                        self.model.set_weights(arrays)
                        logger.info("Initial weights set from PS")
                    else:
                        logger.warning("Initial weights count mismatch (worker).")
                except Exception:
                    logger.exception("Failed to set initial weights")

            # Optionally request latest global weights (no-op if not present)
            try:
                gw = await self.get_weights(job_id)
                if gw.get("status") == "ok" and gw.get("weights") and gw.get("shapes"):
                    arrays = [np.array(l, dtype=np.float32).reshape(tuple(s)) for l, s in zip(gw["weights"], gw["shapes"])]
                    try:
                        if len(arrays) == len(self.model.get_weights()):
                            self.model.set_weights(arrays)
                            logger.info("Initialized model with global weights version=%s", gw.get("version"))
                    except Exception:
                        logger.exception("Failed to apply global weights")
            except Exception:
                logger.debug("get_weights ignored/failure")

            # Prepare disk cache + queue + prefetcher
            cache = DataCache(job_id=job_id)
            queue: asyncio.Queue = asyncio.Queue(maxsize=16)
            prefetcher = TrainPrefetcher(self, job_id, cache, queue, max_download_concurrency=12)

            # Training loop across epochs (PS coordinates epoch cycling)
            cur_epoch = 1
            best_val = float("inf")
            no_improve = 0
            patience = int(hp.get("early_stopping_patience", 10))
            max_epochs = int(hp.get("epochs", 1))

            logger.info("Starting training loop (job=%s)", job_id)

            while True:
                # fetch first batch synchronously to start quickly
                first_batch = await self.fetch_batch(job_id, "train")
                st = first_batch.get("status")
                logger.info("First fetch status: %s", st)

                if st == "ok":
                    # cache first batch images
                    urls_first = first_batch.get("X", []) or []
                    y_first = first_batch.get("y", []) or []
                    first_paths = await cache.download_urls(urls_first, max_concurrency=12)
                    logger.info("Cached first batch (%d images). Starting prefetcher...", len(first_paths))

                    # start background prefetcher for rest of epoch
                    prefetcher.start()

                    # train on first batch
                    try:
                        loss_val = await self._train_on_cached_batch(job_id, cache, first_paths, y_first)
                        logger.info("Trained on first batch; loss=%.6f", loss_val)
                    except Exception:
                        logger.exception("Error training on first batch")

                    # consume queued batches produced by prefetcher
                    while True:
                        item = await queue.get()
                        # control messages
                        if item.get("control"):
                            ctrl = item.get("control")
                            if ctrl == "epoch_end":
                                logger.info("Reached epoch_end (epoch=%s) via control message", item.get("epoch"))
                                # stop prefetcher task and break to run validation / bookkeeping
                                await prefetcher.stop()
                                break
                            if ctrl == "done":
                                logger.info("Received done control (epoch=%s); finishing training for this worker", item.get("epoch"))
                                await prefetcher.stop()
                                # final validation & final metrics reporting
                                #val_loss, val_acc = await self._eval_split(job_id, "val", cache)
                                #logs = {"loss": float(val_loss), "accuracy": float(val_acc), "val_loss": float(val_loss), "val_accuracy": float(val_acc)}
                                logs = {}
                                await self.report_metrics(job_id, int(item.get("epoch", max_epochs)), logs, scope="epoch")
                                final_logs = {"accuracy": float(val_acc), "precision": 0.0, "recall": 0.0, "auc": 0.5, "f1_score": 0.0}
                                await self.report_metrics(job_id, int(item.get("epoch", max_epochs)), final_logs, scope="final", support=0)
                                # notify PS we're done
                                await self.worker_done(job_id)
                                logger.info("Worker notified PS of done. Exiting training loop.")
                                return
                            # unknown control -> ignore
                            continue

                        # regular training batch
                        paths = item.get("paths", []) or []
                        y = item.get("y", []) or []
                        try:
                            loss_val = await self._train_on_cached_batch(job_id, cache, paths, y)
                            logger.info("Trained on cached batch; loss=%.6f", loss_val)
                        except Exception:
                            logger.exception("Error training on cached batch")
                        # no queue.task_done for asyncio.Queue
                        # continue to next item

                elif st == "epoch_end":
                    # no training batches this epoch (rare); treat as epoch boundary
                    logger.info("First fetch returned epoch_end -> no training batches this epoch")
                elif st == "done":
                    # nothing to train; finalize
                    logger.info("First fetch returned done -> finalizing worker")
                    #val_loss, val_acc = await self._eval_split(job_id, "val", cache)
                    #logs = {"loss": float(val_loss), "accuracy": float(val_acc), "val_loss": float(val_loss), "val_accuracy": float(val_acc)}
                    logs = {}
                    await self.report_metrics(job_id, int(first_batch.get("epoch", max_epochs)), logs, scope="epoch")
                    #final_logs = {"accuracy": float(val_acc), "precision": 0.0, "recall": 0.0, "auc": 0.5, "f1_score": 0.0}
                    final_logs = {}
                    await self.report_metrics(job_id, int(first_batch.get("epoch", max_epochs)), final_logs, scope="final", support=0)
                    await self.worker_done(job_id)
                    logger.info("Worker finished with 'done' from PS")
                    return
                else:
                    logger.info("First fetch returned status=%s - backing off briefly", st)
                    await asyncio.sleep(1.0)
                    continue

                # End-of-epoch operations (validation & early stopping)
                logger.info("Epoch %d completed locally — running validation/metrics", cur_epoch)
                #val_loss, val_acc = await self._eval_split(job_id, "val", cache)
                # optionally compute train metrics by re-fetching train (keeps original behavior)
                #train_loss, train_acc = await self._eval_split(job_id, "train", cache)

                #logs = {
                #    "loss": float(train_loss),
                #    "accuracy": float(train_acc),
                #    "val_loss": float(val_loss),
                #    "val_accuracy": float(val_acc)
                #}
                logs = {}
                await self.report_metrics(job_id, cur_epoch, logs, scope="epoch")
                logger.info("Reported epoch %d metrics: %s", cur_epoch, logs)

                # early stopping
                if val_loss < best_val - 1e-8:
                    best_val = val_loss
                    no_improve = 0
                else:
                    no_improve += 1

                cur_epoch += 1
                if no_improve >= patience or cur_epoch > max_epochs:
                    logger.info("Stopping training on worker (early stop or reached max epochs). cur_epoch=%d", cur_epoch)
                    await self.worker_done(job_id)
                    break

            # end training main loop for this job
            logger.info("Worker training loop ended for job %s", job_id)

        finally:
            try:
                await self.socket.close()
            except Exception:
                pass
            try:
                await self.ctx.term()
            except Exception:
                pass
            logger.info("Worker stopped")



def get_worker(cluster_id: int) -> Optional[ClusterNode]:
    ip_address = settings.WORKER_HOST

    db: Session = SessionLocal()
    try:
        return (
            db.query(ClusterNode)
            .filter(
                ClusterNode.cluster_id == cluster_id,
                ClusterNode.node_type == "worker",
                ClusterNode.ip_address == ip_address
            )
            .first()
        )
    finally:
        db.close()

 

async def worker_main(ps_addr: str, worker_id: Optional[str]):
    """Persistent worker process that stays connected and polls for jobs."""
    logger.info(f"Starting worker_main(ps_addr={ps_addr}, worker_id={worker_id})")

    # Re-create a fresh WorkerZMQ instance for each run() to avoid using closed sockets/contexts.
    while True:
        worker = WorkerZMQ(server_addr=ps_addr, worker_id=worker_id)
        try:
            await worker.run()
        except Exception:
            logger.exception("Worker crashed unexpectedly; restarting after 5s")
            await asyncio.sleep(5)
        else:
            logger.info("No active jobs right now; polling again in 10s")
            await asyncio.sleep(10)


def start_worker():
    """Entry point for launching the worker process."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ps", default=settings.PARAMETER_SERVER_URL, help="Parameter server address")
    parser.add_argument("--id", default=None, help="Worker ID")
    args = parser.parse_args()

    asyncio.run(worker_main(args.ps, args.id))


if __name__ == "__main__":
    start_worker()
