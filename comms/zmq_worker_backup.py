"""
ZMQ-based worker-side client to talk to PS.

- Uses DEALER socket to connect to PS ROUTER
- Sends register, heartbeat, send_weights, get_weights messages
- Async via zmq.asyncio

Example usage:
    w = PSWorkerClient(worker_id='worker-1', ps_addr='tcp://192.168.1.10:5555')
    await w.connect()
    await w.register()
    await w.send_weights(job_id, step, weights)
    resp = await w.get_weights(job_id)

This client intentionally keeps payloads JSON-serializable by using lists for tensors.
For production you may switch to binary blobs or compression.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, Any

import zmq
import zmq.asyncio

from utils.logger import get_logger

logger = get_logger("zmq-worker")


class PSWorkerClient:
    def __init__(self, worker_id: str, ps_addr: str = "tcp://127.0.0.1:5555"):
        self.worker_id = worker_id
        self.ps_addr = ps_addr
        self.ctx = zmq.asyncio.Context.instance()
        self.socket = self.ctx.socket(zmq.DEALER)
        # set identity so PS ROUTER can map replies; identity must be bytes
        self.socket.setsockopt(zmq.IDENTITY, worker_id.encode("utf-8"))
        self._lock = asyncio.Lock()

    async def connect(self):
        self.socket.connect(self.ps_addr)
        logger.info("Worker %s connected to PS at %s", self.worker_id, self.ps_addr)

    async def close(self):
        self.socket.close()
        self.ctx.term()

    async def _send_and_recv(self, obj: Dict[str, Any], timeout: float = 5.0) -> Dict[str, Any]:
        async with self._lock:

            payload = json.dumps(obj).encode("utf-8")
            print("------------------------------")
            print(f"Payload: {payload.decode('utf-8')}")
            print("------------------------------")
            await self.socket.send(payload)
            poller = zmq.asyncio.Poller()
            poller.register(self.socket, zmq.POLLIN)
            socks = dict(await poller.poll(timeout * 1000))
            if self.socket in socks:
                parts = await self.socket.recv_multipart()
                if not parts:
                    return {"status": "error", "error": "empty_reply"}
                msg = parts[-1]  # last part contains the actual payload
                try:
                    print("------------------------------")
                    print(f"Received message: {msg.decode("utf-8")}")
                    print("------------------------------")


                    # decode and parse JSON response
                    return json.loads(msg.decode("utf-8"))
                except Exception:
                    logger.exception("Invalid JSON reply from PS")
                    return {"status": "error", "error": "invalid_json_reply"}
            else:
                return {"status": "error", "error": "timeout"}

    async def register(self):
        obj = {"type": "register", "worker_id": self.worker_id}
        return await self._send_and_recv(obj)

    async def heartbeat(self):
        obj = {"type": "heartbeat", "worker_id": self.worker_id}
        return await self._send_and_recv(obj)

    async def send_weights(self, job_id: str, step: int, weights: Dict[str, Any], base_version: int = -1):
        obj = {"type": "send_weights", "worker_id": self.worker_id, "job_id": job_id, "step": step, "weights": weights, "base_version": base_version}
        return await self._send_and_recv(obj)

    async def get_weights(self, job_id: str):
        obj = {"type": "get_weights", "worker_id": self.worker_id, "job_id": job_id}
        return await self._send_and_recv(obj)


# small demo
async def demo():
    client = PSWorkerClient(worker_id="worker-1")
    await client.connect()
    print(await client.register())
    print(await client.heartbeat())
    print(await client.get_weights("job-123"))
    await client.close()


if __name__ == "__main__":
    asyncio.run(demo())
