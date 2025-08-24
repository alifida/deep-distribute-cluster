"""
ZMQ-based Parameter Server (PS) component.

- Uses ROUTER socket to talk to many workers (DEALER side)
- Asyncio-compatible via zmq.asyncio
- Message format: JSON with `type` field, possible types:
    - register: { type: 'register', worker_id }
    - heartbeat: { type: 'heartbeat', worker_id }
    - send_weights: { type: 'send_weights', job_id, step, weights: {layer: [..], ...}, base_version }
    - get_weights: { type: 'get_weights', job_id }

- Replies are JSON with fields: { status: 'ok'|'error', ... }

This module provides a `ParameterServerZMQ` class with:
- start() and stop() async methods
- an in-memory registry for workers and jobs (can be replaced by DB hooks)
- simple aggregation (averaging) of received weight updates per job/step

Note: This is a building block — it intentionally keeps concerns separated so you can
hook it into your existing PS manager / aggregator / db reporter module.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, Any, List

import zmq
import zmq.asyncio

from utils.logger import get_logger
from utils.model_utils import deserialize_weights, serialize_weights, average_weights

logger = get_logger("zmq-ps")


class ParameterServerZMQ:
    def __init__(self, bind_addr: str = "tcp://0.0.0.0:5555"):
        self.bind_addr = bind_addr
        self.ctx = zmq.asyncio.Context.instance()
        self.socket = self.ctx.socket(zmq.ROUTER)
        self._task: asyncio.Task | None = None

        # simple in-memory state
        self.workers: Dict[str, Dict[str, Any]] = {}
        # job_id -> step -> list of weight dicts
        self._recv_buffer: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
        # job_id -> latest global weights (serialized)
        self.global_weights: Dict[str, Dict[str, Any]] = {}
        # job_id -> latest version int
        self.versions: Dict[str, int] = {}

        # aggregation params
        self.quorum_fraction = 0.6
        self.round_timeout = 0.1  # seconds

    async def start(self):
        logger.info("ZMQ PS binding to %s", self.bind_addr)
        self.socket.bind(self.bind_addr)
        self._task = asyncio.create_task(self._serve())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.socket.close()
        self.ctx.term()
        logger.info("ZMQ PS stopped")

    async def _serve(self):
        logger.info("ZMQ PS serving loop started")
        poller = zmq.asyncio.Poller()
        poller.register(self.socket, zmq.POLLIN)

        while True:
            events = dict(await poller.poll(timeout=1000))
            if self.socket in events:
                try:
                    # ROUTER gives [identity, empty, payload]
                    msg_parts = await self.socket.recv_multipart()
                    if len(msg_parts) < 2:
                        logger.warning("Malformed ZMQ message")
                        continue
                    identity = msg_parts[0]
                    payload = msg_parts[-1]
                    await self._handle_message(identity, payload)
                except Exception:
                    logger.exception("Error handling ZMQ message")

            # periodic aggregation: check buffered updates and aggregate by timeout
            await self._process_aggregation_rounds()

    async def _handle_message(self, identity: bytes, raw: bytes):
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            logger.exception("Invalid JSON from worker")
            await self._reply(identity, {"status": "error", "error": "invalid_json"})
            return

        typ = data.get("type")
        if typ == "register":
            worker_id = data.get("worker_id")
            self.workers[worker_id] = {"identity": identity, "last_seen": time.time()}
            logger.info("Worker registered: %s", worker_id)
            await self._reply(identity, {"status": "ok", "msg": "registered"})
            return

        if typ == "heartbeat":
            worker_id = data.get("worker_id")
            record = self.workers.get(worker_id)
            if record:
                record["last_seen"] = time.time()
            await self._reply(identity, {"status": "ok"})
            return

        if typ == "send_weights":
            job_id = data.get("job_id")
            step = int(data.get("step", 0))
            base_version = int(data.get("base_version", -1))
            weights_ser = data.get("weights")  # expected dict of lists
            # store in recv buffer
            self._recv_buffer.setdefault(job_id, {}).setdefault(step, []).append(weights_ser)
            logger.debug("Received weights for job %s step %s from %s", job_id, step, data.get("worker_id"))
            await self._reply(identity, {"status": "ok"})
            return

        if typ == "get_weights":
            job_id = data.get("job_id")
            gw = self.global_weights.get(job_id)
            if gw is None:
                await self._reply(identity, {"status": "error", "error": "weights_not_ready"})
            else:
                await self._reply(identity, {"status": "ok", "weights": gw, "version": self.versions.get(job_id, 0)})
            return

        # Unknown type
        await self._reply(identity, {"status": "error", "error": "unknown_type"})

    async def _reply(self, identity: bytes, obj: Dict[str, Any]):
        try:
            payload = json.dumps(obj).encode("utf-8")
            await self.socket.send_multipart([identity, b"", payload])
        except Exception:
            logger.exception("Failed to reply to worker")

    async def _process_aggregation_rounds(self):
        # For each job and step, if there are buffered updates and quorum reached or timeout elapsed,
        # aggregate and update global_weights
        now = time.time()
        for job_id, steps in list(self._recv_buffer.items()):
            for step, updates in list(steps.items()):
                # simple policy: aggregate when we have >=1 update (you may change to quorum)
                if len(updates) >= 1:
                    try:
                        # deserialize
                        des = [deserialize_weights(u) for u in updates]
                        agg = average_weights(des)
                        ser = serialize_weights(agg)
                        # bump version
                        v = self.versions.get(job_id, 0) + 1
                        self.versions[job_id] = v
                        self.global_weights[job_id] = ser
                        # clear buffer for this step
                        del self._recv_buffer[job_id][step]
                        logger.info("Aggregated weights for job %s step %s -> version %s", job_id, step, v)
                    except Exception:
                        logger.exception("Aggregation failed for job %s step %s", job_id, step)


# Simple runnable helper for manual testing
async def run_server():
    ps = ParameterServerZMQ(bind_addr="tcp://0.0.0.0:5555")
    await ps.start()
    try:
        while True:
            await asyncio.sleep(10)
    except KeyboardInterrupt:
        await ps.stop()


if __name__ == "__main__":
    asyncio.run(run_server())
