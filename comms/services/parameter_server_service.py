# comms/services/parameter_server_service.py
import json
import logging
import zmq
import asyncio

logger = logging.getLogger("parameter-server")

class ParameterServerService:
    def __init__(self, zmq_context, bind_address):
        self.ctx = zmq_context
        self.bind_address = bind_address
        self.socket = self.ctx.socket(zmq.ROUTER)
        self.socket.bind(bind_address)
        logger.info(f"Parameter Server bound to {bind_address}")

    async def listen(self, handler):
        while True:
            parts = await self.socket.recv_multipart()
            if len(parts) < 2:
                logger.error("Invalid message framing: %s", parts)
                continue
            client_id, raw_msg = parts[0], parts[-1]
            try:
                msg = json.loads(raw_msg.decode())
            except json.JSONDecodeError:
                logger.error("Invalid JSON message: %s", raw_msg)
                continue

            logger.info("Received: %s", msg)
            response = await handler(msg)
            await self.socket.send_multipart([client_id, json.dumps(response).encode()])

    async def send_message(self, client_id, msg):
        await self.socket.send_multipart([client_id, json.dumps(msg).encode()])
