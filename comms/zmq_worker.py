
import asyncio
import json
import logging
import uuid
import io
import random
from typing import List, Tuple

import aiohttp
import numpy as np
import zmq.asyncio
from PIL import Image
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("worker")

# ---------- Utility to download/resize images ----------
async def download_and_preprocess(session: aiohttp.ClientSession, url: str, target_size: Tuple[int, int]) -> np.ndarray:
    try:
        async with session.get(url, timeout=30) as resp:
            if resp.status != 200:
                raise RuntimeError(f"bad status {resp.status} for {url}")
            b = await resp.read()
        img = Image.open(io.BytesIO(b)).convert("RGB")
        img = img.resize(target_size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr
    except Exception as e:
        logger.error("Failed to download/process %s: %s", url, e)
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.float32)



# ---------- Helpers to serialize weights ----------
def serialize_weights(model_weights: List[np.ndarray]):
    shapes = [w.shape for w in model_weights]
    lists = [w.flatten().astype(float).tolist() for w in model_weights]
    return lists, shapes

def deserialize_weights(lists: List[List[float]], shapes: List[Tuple[int, ...]]):
    arrays = []
    for lst, shape in zip(lists, shapes):
        arr = np.array(lst, dtype=np.float32).reshape(shape)
        arrays.append(arr)
    return arrays

def model_from_json(json_str: str) -> tf.keras.Model:
    """Rebuild model from JSON architecture string."""
    return tf.keras.models.model_from_json(json_str)
# ---------- Optimizer / loss factories ----------
def make_optimizer(name: str, lr: float):
    name = (name or "adam").lower()
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    if name == "adam" or name == "adamw":
        # keep Adam for both keys for simplicity
        return tf.keras.optimizers.Adam(learning_rate=lr)
    return tf.keras.optimizers.Adam(learning_rate=lr)

def make_loss(name: str):
    name = (name or "binary_crossentropy").lower()
    if name in ("binary_crossentropy", "bce"):
        return tf.keras.losses.BinaryCrossentropy(from_logits=False)
    if name in ("categorical_crossentropy", "cce"):
        # our small demo is binary; mapping to BCE for now
        return tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return tf.keras.losses.BinaryCrossentropy(from_logits=False)

# ---------- Worker Class ----------
class WorkerZMQ:
    def __init__(self, server_addr="tcp://127.0.0.1:5555", worker_id=None, image_size=(224, 224)):
        self.server_addr = server_addr
        self.worker_id = worker_id or str(uuid.uuid4())
        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.IDENTITY, self.worker_id.encode())
        self.running = False
        self.image_size = image_size

        # Will be set after registering to a job
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.hyperparams = None

    async def start(self):
        self.socket.connect(self.server_addr)
        logger.info("Worker %s connected to %s", self.worker_id, self.server_addr)
        self.running = True

    async def stop(self):
        self.running = False
        self.socket.close()
        self.ctx.term()
        logger.info("Worker %s stopped", self.worker_id)

    async def _send_and_recv(self, message: dict, timeout: float = 20.0) -> dict:
        try:
            await self.socket.send_multipart([b"", json.dumps(message).encode()])
            parts = await asyncio.wait_for(self.socket.recv_multipart(), timeout=timeout)
            reply = parts[-1]
            return json.loads(reply.decode())
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for server response")
            return {"status": "error", "error": "timeout"}
        except Exception as e:
            logger.exception("Error send/recv: %s", e)
            return {"status": "error", "error": str(e)}

    async def register(self, job_id: str):
        msg = {"type": "register", "worker_id": self.worker_id, "job_id": job_id}
        return await self._send_and_recv(msg)

    async def fetch_batch(self, job_id: str, set_name: str = "train"):
        msg = {"type": "get_batch", "worker_id": self.worker_id, "job_id": job_id, "set": set_name}
        return await self._send_and_recv(msg)

    async def send_weights(self, job_id: str, lists, shapes):
        msg = {"type": "send_weights", "job_id": job_id, "worker_id": self.worker_id,
               "weights": lists, "shapes": shapes}
        return await self._send_and_recv(msg)

    async def get_weights(self, job_id: str):
        msg = {"type": "get_weights", "job_id": job_id, "worker_id": self.worker_id}
        return await self._send_and_recv(msg)

    async def report_metrics(self, job_id: str, epoch: int, logs: dict, scope: str = "epoch", support: int = 0):
        msg = {
            "type": "report_metrics",
            "job_id": job_id,
            "worker_id": self.worker_id,
            "epoch": epoch,
            "scope": scope,         # "epoch" or "final"
            "logs": logs,
            "support": support
        }
        return await self._send_and_recv(msg)

    async def worker_done(self, job_id: str):
        msg = {"type": "worker_done", "job_id": job_id, "worker_id": self.worker_id}
        return await self._send_and_recv(msg)

    # --------- Training utilities ----------
    async def _download_stack(self, urls: List[str]) -> np.ndarray:
        async with aiohttp.ClientSession() as session:
            imgs = await asyncio.gather(*(download_and_preprocess(session, url, self.image_size) for url in urls))
        return np.stack(imgs, axis=0)

    def _apply_hyperparams(self, hp: dict):
        # seeds
        seed = int(hp.get("random_seed", 42))
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)

        # rebuild model with dropout + optimizer/loss
        #img_h, img_w = hp.get("image_size", (self.image_size[0], self.image_size[1]))
        #self.model = build_model(input_shape=(img_h, img_w, 3), dropout_rate=float(hp.get("dropout_rate", 0.0)))
        #self.optimizer = make_optimizer(hp.get("optimizer"), float(hp.get("learning_rate", 1e-3)))
        #self.loss_fn = make_loss(hp.get("loss_function"))
        self.hyperparams = hp
        # Prepare optimizer and loss
        self.optimizer = make_optimizer(hp.get("optimizer"), float(hp.get("learning_rate", 1e-3)))
        self.loss_fn = make_loss(hp.get("loss_function"))

    async def train_on_batch(self, job_id: str, X_urls: List[str], y_labels: List[int]) -> float:
        X = await self._download_stack(X_urls)
        y = np.array(y_labels, dtype=np.float32)

        X_tensor = tf.convert_to_tensor(X)
        y_tensor = tf.convert_to_tensor(y)

        with tf.GradientTape() as tape:
            preds = tf.reshape(self.model(X_tensor, training=True), [-1])
            loss = self.loss_fn(y_tensor, preds)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # push weights immediately so others can pick latest
        lists, shapes = serialize_weights(self.model.get_weights())
        await self.send_weights(job_id, lists, shapes)
        return float(loss)

    async def eval_full_split(self, job_id: str, set_name: str):
        """Iterate through val/test split and compute simple metrics."""
        loss_sum, n, correct = 0.0, 0, 0
        while True:
            batch = await self.fetch_batch(job_id, set_name)
            st = batch.get("status")
            if st != "ok":
                break
            X = await self._download_stack(batch["X"])
            y = np.array(batch["y"], dtype=np.float32)
            preds = tf.reshape(self.model(tf.convert_to_tensor(X), training=False), [-1]).numpy()
            # loss
            loss = self.loss_fn(tf.convert_to_tensor(y), tf.convert_to_tensor(preds)).numpy().item()
            loss_sum += loss * len(y)
            # accuracy (threshold 0.5)
            correct += int(np.sum((preds >= 0.5).astype(np.float32) == y))
            n += len(y)
        avg_loss = (loss_sum / n) if n > 0 else 0.0
        acc = (correct / n) if n > 0 else 0.0
        return avg_loss, acc

    # ---------- Main loop ----------
    async def run(self):
        await self.start()
        try:
            while self.running:
                # Ask PS for available jobs
                jobs_resp = await self._send_and_recv({"type": "get_available_jobs", "worker_id": self.worker_id})
                job_ids = jobs_resp.get("job_ids", [])
                if not job_ids:
                    await asyncio.sleep(3)
                    continue

                for job_id in job_ids:
                    reg = await self.register(job_id)
                    if reg.get("status") != "ok":
                        logger.error("Registration failed for job %s: %s", job_id, reg)
                        continue


                    hp = reg.get("hyperparams", {}) or {}
                    model_json = reg.get("model_json")
                    init_w = reg.get("initial_weights", {})
                    self._apply_hyperparams(hp)

                    # Build model from PS JSON
                    if not model_json:
                        logger.error("No model_json received from PS for job %s", job_id)
                        continue

                    try:
                        self.model = model_from_json(model_json)
                        logger.info("Worker %s built model from PS JSON", self.worker_id)
                    except Exception as e:
                        logger.exception("Failed to build model from JSON: %s", e)
                        continue

                    # Load initial weights
                    if init_w.get("lists") and init_w.get("shapes"):
                        try:
                            arrays = deserialize_weights(init_w["lists"], init_w["shapes"])
                            self.model.set_weights(arrays)
                            logger.info("Worker %s set initial weights from PS", self.worker_id)
                        except Exception as e:
                            logger.exception("Failed to set initial weights: %s", e)





                    # Train over epochs (PS cycles epochs by replying `epoch_end`/`done`)
                    cur_epoch = 1
                    best_val = float("inf")
                    no_improve = 0
                    patience = int(hp.get("early_stopping_patience", 10))
                    epochs = int(hp.get("epochs", 1))

                    while self.running:
                        batch = await self.fetch_batch(job_id, "train")
                        st = batch.get("status")
                        logger.info(f"----{st}---")
                        if st == "ok":
                            loss = await self.train_on_batch(job_id, batch["X"], batch["y"])
                            # (Optional) could report per-batch metrics if you want:
                            # await self.report_metrics(job_id, cur_epoch, {"batch_loss": loss}, scope="batch")
                            continue

                        if st == "epoch_end":
                            # validation pass at the end of epoch
                            val_loss, val_acc = await self.eval_full_split(job_id, "val")

                            # Always compute train loss/accuracy on full training set, not just this batch
                            train_loss, train_acc = await self.eval_full_split(job_id, "train")

                            logs = {
                                "loss": float(train_loss) if train_loss is not None else 0.0,
                                "accuracy": float(train_acc) if train_acc is not None else 0.0,
                                "val_loss": float(val_loss) if val_loss is not None else 0.0,
                                "val_accuracy": float(val_acc) if val_acc is not None else 0.0
                            }
                            await self.report_metrics(job_id, cur_epoch, logs, scope="epoch")
                            # early stopping check
                            if val_loss < best_val - 1e-8:
                                best_val = val_loss
                                no_improve = 0
                            else:
                                no_improve += 1
                            cur_epoch += 1
                            if no_improve >= patience:
                                logger.info("Early stopping triggered on worker %s at epoch %d", self.worker_id, cur_epoch - 1)
                                break
                            continue

                        if st == "done":
                            # final validation on completion
                            val_loss, val_acc = await self.eval_full_split(job_id, "val")

                            if "X" in batch and "y" in batch and len(batch["X"]) > 0:
                                loss, accuracy = self.model.evaluate(batch["X"], batch["y"], verbose=0)
                            else:
                                loss, accuracy = val_loss, val_acc  # fallback if no batch data

                            logs = {
                                "loss": float(loss) if loss is not None else 0.0,
                                "accuracy": float(accuracy) if accuracy is not None else 0.0,
                                "val_loss": float(val_loss) if val_loss is not None else 0.0,
                                "val_accuracy": float(val_acc) if val_acc is not None else 0.0
                            }
                            await self.report_metrics(job_id, int(batch.get("epoch", epochs)), logs, scope="epoch")
                            
                            # Build final metrics with support (number of samples used)
                            if "X" in batch and "y" in batch and len(batch["y"]) > 0:
                                y_true = np.array(batch["y"]).astype(int)
                                y_pred = (self.model.predict(batch["X"]) > 0.5).astype(int).ravel()
                                try:
                                    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
                                    precision = float(precision_score(y_true, y_pred, zero_division=0))
                                    recall    = float(recall_score(y_true, y_pred, zero_division=0))
                                    f1        = float(f1_score(y_true, y_pred, zero_division=0))
                                    auc       = float(roc_auc_score(y_true, y_pred))
                                except Exception:
                                    precision, recall, f1, auc = 0.0, 0.0, 0.0, 0.5
                                support = int(len(y_true))
                            else:
                                precision = recall = f1 = 0.0
                                auc = 0.5
                                support = 0

                            final_logs = {
                                "accuracy": float(accuracy),
                                "precision": precision,
                                "recall": recall,
                                "auc": auc,
                                "f1_score": f1
                            }
                            # report final metrics to PS (so PS can aggregate and persist)
                            await self.report_metrics(job_id, int(batch.get("epoch", epochs)), final_logs, scope="final", support=support)
                            # notify PS this worker is fully done
                            await self._send_and_recv({"type": "worker_done", "job_id": job_id, "worker_id": self.worker_id})
                            
                            
                            break

                        if st == "empty":
                            # nothing right now (val/test exhausted)
                            await asyncio.sleep(1)
                            continue

                        # unknown status -> short backoff
                        await asyncio.sleep(1)

                    await self.worker_done(job_id)

                await asyncio.sleep(2)
        finally:
            await self.stop()

# Entry point
if __name__ == "__main__":
    worker = WorkerZMQ()
    asyncio.run(worker.run())
