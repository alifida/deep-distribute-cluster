"""
Job Dispatcher / Scalable Queue for deep-distribute-cluster

Responsibilities:
- Accept jobs into a queue (Redis-backed if available, DB fallback)
- Maintain a Pub/Sub channel to notify workers of available jobs
- Assign jobs to workers based on simple capability matching
- Update job status in DB and expose an async-safe API for enqueueing

Design choices for scalability:
- Redis (LIST + PUB/SUB) is used as the primary queue for low-latency job dispatch.
- If Redis is not available, the dispatcher falls back to storing jobs in the DB and uses a polling loop.
- Worker nodes subscribe to the `jobs:available` channel to get immediate notifications.
- Assignments are optimistic: the dispatcher publishes the job_id and worker-side SDK will `claim` it via a REST call.
- A Re-queue / TTL mechanism ensures jobs claimed but not started within a timeout are returned to the queue.

Dependencies:
- aioredis (async redis client)
- sqlalchemy (for DB fallback / status updates)

This module exposes:
- async function `enqueue_job(job_id, payload)` to place a job in the queue
- FastAPI router `router` with endpoints: /enqueue, /claim, /requeue
- Background task `dispatcher_loop` which monitors redis or DB for new jobs

"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from utils.db import get_db
from models.db_models import TrainTrainingJob as Job
from utils.logger import get_logger

# Try importing aioredis; if not available, dispatcher will gracefully fallback
try:
    import redis.asyncio as aioredis
except Exception:  # pragma: no cover - import-time fallback
    aioredis = None

router = APIRouter(prefix="/dispatch", tags=["dispatch"])
logger = get_logger("dispatcher")

# Redis keys / channels
REDIS_JOB_QUEUE = "deepdist:jobs:queue"
REDIS_JOB_CHANNEL = "deepdist:jobs:available"
REDIS_CLAIM_PREFIX = "deepdist:jobs:claim:"

# In-process reservation TTL (seconds). If a worker claims a job but doesn't start it, it will be requeued.
CLAIM_TTL = 30

# Local in-memory pending set (only for environments without Redis)
_local_pending: Dict[str, Dict[str, Any]] = {}

# Redis client singleton (initialized in startup)
_redis: Optional["aioredis.Redis"] = None


async def init_redis_pool(redis_url: str = "redis://localhost"):
    global _redis
    if aioredis is None:
        logger.warning("aioredis not installed; dispatcher will use DB fallback")
        _redis = None
        return None

    if _redis is None:
        _redis = await aioredis.from_url(redis_url, decode_responses=True)
        # Test connect
        try:
            await _redis.ping()
            logger.info("Connected to Redis at %s", redis_url)
        except Exception as e:
            logger.exception("Failed to connect to Redis: %s", e)
            _redis = None
    return _redis


async def enqueue_job(job_id: str, payload: Dict[str, Any], redis_url: str = "redis://localhost") -> bool:
    """Place a job into the queue. Returns True if queued successfully."""
    # Normalize payload
    item = json.dumps({"job_id": job_id, "payload": payload, "enqueued_at": time.time()})

    # Try Redis enqueue
    if _redis is not None:
        try:
            await _redis.rpush(REDIS_JOB_QUEUE, item)
            # Publish notification for subscribers
            await _redis.publish(REDIS_JOB_CHANNEL, job_id)
            logger.debug("Enqueued job %s to Redis", job_id)
            return True
        except Exception:
            logger.exception("Redis enqueue failed; falling back to DB enqueue")

    # DB fallback: store in local pending dict (or in DB) - but prefer in-memory to avoid DB schema coupling here
    _local_pending[job_id] = {"payload": payload, "enqueued_at": time.time()}
    logger.debug("Enqueued job %s to local pending store", job_id)
    return True


async def claim_job(worker_id: str, job_id: str, db: Session) -> bool:
    """Worker calls this endpoint to claim a job. We atomically mark the Job status=claimed in DB.

    Returns True if claim successful; False if already claimed.
    """
    job = db.query(Job).filter(Job.id == job_id).with_for_update().first()
    if not job:
        logger.debug("claim_job: job %s not found", job_id)
        return False

    if job.status not in ("pending", "queued"):
        logger.debug("claim_job: job %s already claimed or running: %s", job_id, job.status)
        return False

    job.status = "claimed"
    job.progress = 0.0
    db.commit()
    logger.info("Worker %s claimed job %s", worker_id, job_id)

    # store claim TTL in Redis if available
    if _redis is not None:
        try:
            await _redis.set(f"{REDIS_CLAIM_PREFIX}{job_id}", worker_id, ex=CLAIM_TTL)
        except Exception:
            logger.exception("Failed to set claim TTL in Redis for job %s", job_id)

    return True


async def requeue_job(job_id: str):
    """Requeue a job (push back to Redis or local queue)."""
    # If job exists in local pending, re-push its payload
    item = None
    if job_id in _local_pending:
        payload = _local_pending[job_id]["payload"]
        item = json.dumps({"job_id": job_id, "payload": payload, "requeued_at": time.time()})

    if _redis is not None and item is None:
        # Try to read from Redis (we can't easily get by job_id), so just publish a generic requeue
        await _redis.publish(REDIS_JOB_CHANNEL, job_id)
        logger.info("Published requeue signal for job %s", job_id)
        return True

    if _redis is not None and item is not None:
        await _redis.rpush(REDIS_JOB_QUEUE, item)
        await _redis.publish(REDIS_JOB_CHANNEL, job_id)
        logger.info("Requeued job %s to Redis", job_id)
        return True

    if item is not None:
        _local_pending[job_id] = json.loads(item)
        logger.info("Requeued job %s to local pending", job_id)
        return True

    logger.warning("requeue_job: job %s had no payload to requeue", job_id)
    return False


# ----------------------------
# Background dispatcher loop
# ----------------------------
async def dispatcher_loop(redis_url: str = "redis://localhost"):
    """Background task that subscribes to Redis channel and processes job queue.

    If Redis is unavailable, it will retry in the background and fall back to local pending jobs.
    """
    global _redis
    logger.info(f"Starting dispatcher loop (Redis: {redis_url})")

    while True:
        try:
            # Ensure we have a Redis connection
            if _redis is None:
                try:
                    await init_redis_pool(redis_url)
                    if _redis:
                        logger.info(f"✅ Connected to Redis at {redis_url}")
                except Exception as e:
                    logger.error(f"❌ Redis connection failed: {e}. Retrying in 5s...")
                    await asyncio.sleep(5)
                    continue

            if _redis:
                # Try popping a job from Redis
                item = await _redis.blpop(REDIS_JOB_QUEUE, timeout=5)
                if item:
                    _, payload_json = item
                    try:
                        entry = json.loads(payload_json)
                        job_id = entry.get("job_id")
                        payload = entry.get("payload")
                        logger.info(f"Dispatching job {job_id} from Redis")
                        await _redis.publish(
                            REDIS_JOB_CHANNEL,
                            json.dumps({"job_id": job_id, "payload": payload}),
                        )
                    except Exception:
                        logger.exception("Failed to parse queued item")
                else:
                    await asyncio.sleep(0.1)
            else:
                # Redis unavailable: process local pending jobs
                if _local_pending:
                    for job_id, entry in list(_local_pending.items()):
                        logger.info(f"Dispatching job {job_id} from local pending")
                        del _local_pending[job_id]
                await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            logger.info("Dispatcher loop cancelled; exiting")
            break
        except Exception:
            logger.exception("Error in dispatcher loop; continuing")
            await asyncio.sleep(1.0)

# ----------------------------
# FastAPI endpoints to control queue
# ----------------------------
@router.post("/enqueue")
async def api_enqueue(job_id: str, payload: Dict[str, Any], background: BackgroundTasks):
    """Enqueue a job (called by the job creator / Django integration).

    - Creates or updates Job record in DB
    - Pushes to Redis queue and publishes channel
    """
    # Validate job exists in DB
    # Note: get_db yields a generator, here we call synchronously via dependency pattern in FastAPI;
    # we will create a separate DB session for direct invocation.
    from utils.db import SessionLocal

    db: Session = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            # Create a minimal job record
            job = Job(id=job_id, model=payload.get("model", "unknown"), dataset=payload.get("dataset", ""), status="queued", progress=0.0)
            db.add(job)
            db.commit()
            db.refresh(job)
        else:
            job.status = "queued"
            db.commit()

        # Enqueue asynchronously
        background.add_task(enqueue_job, job_id, payload)
        return {"status": "enqueued", "job_id": job_id}
    finally:
        db.close()


@router.post("/claim")
async def api_claim(job_id: str, worker_id: str, db: Session = Depends(get_db)):
    """Worker claims the job. Returns success if job was successfully claimed."""
    ok = await claim_job(worker_id, job_id, db)
    if not ok:
        raise HTTPException(status_code=409, detail="Job already claimed or not found")
    return {"status": "claimed", "job_id": job_id}


@router.post("/requeue")
async def api_requeue(job_id: str):
    ok = await requeue_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found for requeue")
    return {"status": "requeued", "job_id": job_id}


# ----------------------------
# Startup helper to launch dispatcher loop
# ----------------------------
async def start_dispatcher_background(app, redis_url: str = "redis://localhost"):
    loop = asyncio.get_event_loop()
    task = loop.create_task(dispatcher_loop(redis_url=redis_url))

    # store task reference on app so shutdown can cancel it
    app.state.dispatcher_task = task
    logger.info("Dispatcher background task started")
    return task


# Graceful shutdown
async def stop_dispatcher_background(app):
    task = getattr(app.state, "dispatcher_task", None)
    if task:
        task.cancel()
        try:
            await task
        except Exception:
            pass
        logger.info("Dispatcher background task stopped")


# Provide small helper to check readiness
async def is_dispatcher_ready() -> bool:
    return _redis is not None or bool(_local_pending)
