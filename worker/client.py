"""
Worker Client SDK

This module allows a worker process to:
- Subscribe to job availability notifications
- Claim a job from the dispatcher
- Download dataset shard if required
- Run training loop (stubbed for now)
- Send telemetry and job completion updates
"""

import asyncio
import aiohttp
import redis.asyncio as aioredis
import json
from utils.config import settings
from utils.logger import get_logger

logger = get_logger("worker-client")


class WorkerClient:
    def __init__(self, worker_id: str, api_base: str = "http://localhost:8000"):
        self.worker_id = worker_id
        self.api_base = api_base
        self.redis_url = settings.REDIS_URL
        self.job_channel = settings.JOB_CHANNEL
        self.session = None
        self.redis = None

    async def connect(self):
        self.session = aiohttp.ClientSession()
        self.redis = await aioredis.from_url(self.redis_url, decode_responses=True)
        logger.info(f"Worker {self.worker_id} connected to Redis and API {self.api_base}")

    async def listen_for_jobs(self):
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(self.job_channel)
        logger.info(f"Worker {self.worker_id} subscribed to {self.job_channel}")
        async for message in pubsub.listen():
            if message["type"] == "message":
                logger.info(f"Worker {self.worker_id} received job notification")
                await self.claim_job()

    async def claim_job(self):
        url = f"{self.api_base}/dispatch/claim/{self.worker_id}"
        async with self.session.post(url) as resp:
            if resp.status == 200:
                job = await resp.json()
                job_id = job.get("job_id")
                logger.info(f"Worker {self.worker_id} claimed job {job_id}")
                await self.run_job(job)
            else:
                logger.warning(f"Worker {self.worker_id} failed to claim job: {resp.status}")

    async def run_job(self, job: dict):
        job_id = job.get("job_id")
        dataset_url = job.get("dataset_url")
        logger.info(f"Worker {self.worker_id} starting job {job_id} with dataset {dataset_url}")

        # TODO: implement dataset download & training loop
        await asyncio.sleep(2)  # simulate training

        # Report completion
        url = f"{self.api_base}/jobs/{job_id}/complete"
        async with self.session.post(url, json={"worker_id": self.worker_id}) as resp:
            if resp.status == 200:
                logger.info(f"Worker {self.worker_id} completed job {job_id}")
            else:
                logger.error(f"Worker {self.worker_id} failed to report completion for {job_id}")

    async def send_telemetry(self, job_id: str, metrics: dict):
        url = f"{self.api_base}/telemetry/{job_id}"
        async with self.session.post(url, json={"worker_id": self.worker_id, "metrics": metrics}) as resp:
            if resp.status != 200:
                logger.error(f"Worker {self.worker_id} failed to send telemetry for job {job_id}")

    async def close(self):
        if self.session:
            await self.session.close()
        if self.redis:
            await self.redis.close()


async def main():
    worker = WorkerClient(worker_id="worker-1")
    await worker.connect()
    await worker.listen_for_jobs()


if __name__ == "__main__":
    asyncio.run(main())
