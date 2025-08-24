# app/worker/trainer.py
"""
Worker Training Loop
--------------------
This script runs a training loop that:
1. Loads a dataset (dummy data for now).
2. Computes fake gradients or weight updates.
3. Sends them to the Parameter Server (PS) via ZMQ.
4. Fetches updated global weights from PS.
"""

import asyncio
import logging
import numpy as np
from comms.zmq_worker import PSWorkerClient

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, worker_id: str, ps_url: str, job_id: str):
        self.worker_id = worker_id
        self.ps_url = ps_url
        self.job_id = job_id
        self.client = PSWorkerClient(worker_id=worker_id, ps_addr=ps_url)
        self.global_weights = None

    async def setup(self):
        """Connect to PS and register worker."""
        await self.client.connect()
        logger.info("Connected to PS at %s", self.ps_url)
        await self.client.register()
        logger.info("Worker registered with PS.")

    async def train_step(self, step: int):
        """
        Dummy train step:
        - Simulates computation of gradients.
        - Sends gradients to PS.
        - Retrieves updated global weights.
        """
        logger.info("Step %d: computing fake gradients...", step)
        fake_gradients = np.random.randn(5).tolist()  # Replace with real gradients

        # Send gradients to PS
        send_resp = await self.client.send_weights(self.job_id, step, fake_gradients)
        logger.info("Sent weights: %s", send_resp)

        # Fetch latest global weights
        self.global_weights = await self.client.get_weights(self.job_id)
        logger.info("Fetched global weights: %s", self.global_weights)

    async def run_training(self, steps: int = 5):
        """Run the training loop for N steps."""
        await self.setup()
        for step in range(steps):
            await self.train_step(step)
            await asyncio.sleep(1)  # Simulate compute time
        logger.info("Training completed for job %s", self.job_id)


async def main():
    trainer = Trainer(
        worker_id="worker-1",
        ps_url="tcp://localhost:5555",
        job_id="job-123"
    )
    await trainer.run_training(steps=5)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
