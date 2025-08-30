"""
ZeroMQ client utility for worker <-> parameter server communication.
Handles sending and receiving messages for weight synchronization and control.
"""

import zmq
import json
from typing import Dict, Any, Optional


class ZMQClient:
    def __init__(self, ps_host: str = "127.0.0.1", ps_port: int = 5555):
        self.ps_host = ps_host
        self.ps_port = ps_port
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REQ)
        self.socket.connect(f"tcp://{ps_host}:{ps_port}")

    def send(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            self.socket.send_json(message)
            reply = self.socket.recv_json()
            return reply
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def send_weights(self, job_id: str, step: int, weights: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        msg = {"type": "weights", "job_id": job_id, "step": step, "weights": weights}
        return self.send(msg)

    def request_weights(self, job_id: str) -> Optional[Dict[str, Any]]:
        msg = {"type": "get_weights", "job_id": job_id}
        return self.send(msg)

    def heartbeat(self, worker_id: str) -> Optional[Dict[str, Any]]:
        msg = {"type": "heartbeat", "worker_id": worker_id}
        return self.send(msg)

    def close(self):
        self.socket.close()
        self.ctx.term()
