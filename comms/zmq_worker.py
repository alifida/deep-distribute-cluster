# comms/zmq_worker_dynamic.py
import asyncio
import json
import logging
import uuid
import io
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

# ---------- Simple model builder ----------
def model_from_name(algo_name: str, input_shape=(224, 224, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, 3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)

# ---------- Helpers to serialize weights ----------
def serialize_weights(model_weights: List[np.ndarray]) -> Tuple[List[List[float]], List[Tuple[int, ...]]]:
    shapes = [w.shape for w in model_weights]
    lists = [w.flatten().astype(float).tolist() for w in model_weights]
    return lists, shapes

def deserialize_weights(lists: List[List[float]], shapes: List[Tuple[int, ...]]) -> List[np.ndarray]:
    arrays = []
    for lst, shape in zip(lists, shapes):
        arr = np.array(lst, dtype=np.float32).reshape(shape)
        arrays.append(arr)
    return arrays

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

        # Build model and optimizer
        self.model = model_from_name("tiny", input_shape=(image_size[1], image_size[0], 3))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    async def start(self):
        self.socket.connect(self.server_addr)
        logger.info("Worker %s connected to %s", self.worker_id, self.server_addr)
        self.running = True

    async def stop(self):
        self.running = False
        self.socket.close()
        self.ctx.term()
        logger.info("Worker %s stopped", self.worker_id)

    async def _send_and_recv(self, message: dict, timeout: float = 10.0) -> dict:
        try:
            # Send message as multipart: empty delimiter + payload JSON
            await self.socket.send_multipart([b"", json.dumps(message).encode()])

            # Receive response from server (ROUTER may include extra frames)
            parts = await asyncio.wait_for(self.socket.recv_multipart(), timeout=timeout)

            # The last frame is the JSON payload
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

    async def fetch_batch(self, job_id: str):
        msg = {"type": "get_batch", "worker_id": self.worker_id, "job_id": job_id}
        return await self._send_and_recv(msg)

    async def send_weights(self, job_id: str, lists, shapes):
        msg = {"type": "send_weights", "job_id": job_id, "worker_id": self.worker_id,
               "weights": lists, "shapes": shapes}
        return await self._send_and_recv(msg)

    async def get_weights(self, job_id: str):
        msg = {"type": "get_weights", "job_id": job_id, "worker_id": self.worker_id}
        return await self._send_and_recv(msg)

    async def train_on_batch(self, job_id: str, X_urls: List[str], y_labels: List[int]):
        async with aiohttp.ClientSession() as session:
            imgs = await asyncio.gather(*(download_and_preprocess(session, url, self.image_size) for url in X_urls))

        X = np.stack(imgs, axis=0)
        y = np.array(y_labels, dtype=np.float32)

        X_tensor = tf.convert_to_tensor(X)
        y_tensor = tf.convert_to_tensor(y)

        with tf.GradientTape() as tape:
            preds = tf.reshape(self.model(X_tensor, training=True), [-1])
            loss = self.loss_fn(y_tensor, preds)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # TensorFlow Keras model.get_weights() returns NumPy arrays directly
        weights = self.model.get_weights()
        lists, shapes = serialize_weights(weights)
        resp = await self.send_weights(job_id, lists, shapes)
        logger.info("Worker %s sent weights for job %s, loss=%.5f", self.worker_id, job_id, float(loss))
        return float(loss)

    # ---------- Main loop to continuously poll for jobs ----------
    async def run(self):
        await self.start()
        try:
            while self.running:
                # 1. Ask PS if there are jobs available (here PS can have a /get_jobs endpoint or custom msg)
                msg = {"type": "get_available_jobs", "worker_id": self.worker_id}
                jobs_resp = await self._send_and_recv(msg)
                job_ids = jobs_resp.get("job_ids", [])
                
                if not job_ids:
                    await asyncio.sleep(5)  # no job yet, retry
                    continue

                for job_id in job_ids:
                    reg_resp = await self.register(job_id)
                    if reg_resp.get("status") != "ok":
                        logger.error("Registration failed for job %s: %s", job_id, reg_resp)
                        continue

                    # Fetch global weights if available
                    gw = await self.get_weights(job_id)
                    if gw.get("status") == "ok" and gw.get("weights") and gw.get("shapes"):
                        try:
                            arrays = deserialize_weights(gw["weights"], gw["shapes"])
                            self.model.set_weights(arrays)
                            logger.info("Worker %s initialized model from global weights (v=%s)", self.worker_id, gw.get("version"))
                        except Exception as e:
                            logger.exception("Failed to set model weights: %s", e)
                    else:
                        logger.info("No global weights for job %s, starting fresh", job_id)  
                    # Process batches until done
                    while self.running:
                        batch = await self.fetch_batch(job_id)
                        if batch.get("status") == "ok" and "X" in batch:
                            logger.info("Worker %s received batch for job %s", self.worker_id, job_id)
                            await self.train_on_batch(job_id, batch["X"], batch["y"])
                        elif batch.get("status") == "empty":
                            logger.info("No more batches for job %s", job_id)
                            break
                        else:
                            logger.info("No batch available for job %s, retrying... after 2 sec", job_id)
                            await asyncio.sleep(2)
                
                await asyncio.sleep(3)  # poll interval between job queries

        finally:
            await self.stop()

# Entry point
if __name__ == "__main__":
    worker = WorkerZMQ()
    asyncio.run(worker.run())
