# comms/services/worker_service.py
import json
import logging
import zmq
import asyncio

logger = logging.getLogger("worker-service")

class WorkerService:
    def __init__(self, zmq_context, server_address):
        self.ctx = zmq_context
        self.server_address = server_address
        self.socket = self.ctx.socket(zmq.DEALER)

    async def connect(self, worker_id):
        self.socket.setsockopt(zmq.IDENTITY, worker_id.encode())
        self.socket.connect(self.server_address)
        logger.info(f"Worker connected to {self.server_address} with ID {worker_id}")

    async def send_and_receive(self, message, timeout=10.0):
        try:
            await self.socket.send_json(message)
            parts = await asyncio.wait_for(self.socket.recv_multipart(), timeout=timeout)
            reply = json.loads(parts[-1].decode())
            return reply
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for response")
            return {"status": "error", "error": "timeout"}
        except Exception as e:
            logger.exception("Error send/recv: %s", e)
            return {"status": "error", "error": str(e)}
