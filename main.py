"""
Main FastAPI application entry point.

- Initializes DB
- Registers routers for jobs, workers, datasets, telemetry, dispatcher
- Starts/stops dispatcher background task
"""


from fastapi import FastAPI, HTTPException, Path, Body
from api.routes import jobs as jobs_router
from utils.db import Base, engine
from utils.config import settings
from utils.logger import logger
from api import jobs, workers, datasets, telemetry
from dispatcher import start_dispatcher_background, stop_dispatcher_background
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import zmq
import zmq.asyncio
from config import settings
# Import the parameter server class
from comms.zmq_ps import ParameterServerZMQ

# Create tables
Base.metadata.create_all(bind=engine)

# Init app
app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# Routers
#app.include_router(jobs.router)
#app.include_router(workers.router)
#app.include_router(datasets.router)
#app.include_router(telemetry.router)
#app.include_router(jobs_router.router)



# Create a global instance of Parameter Server
ps = ParameterServerZMQ(bind_addr="tcp://0.0.0.0:5555")


class TrainRequest(BaseModel):
    X: List[List[float]]  # 2D data array
    y: List[float]        # Labels




@app.on_event("startup")
async def startup_event():
    
    """Start the Parameter Server when API boots."""
    logger.info("🚀 Starting Parameter Server...")
    await ps.start()
    
    logger.info("🚀 Starting  dispatcher...")
    await start_dispatcher_background(app)
    logger.info("🚀 STARTED dispatcher...")



@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down FastAPI app and dispatcher...")
    await stop_dispatcher_background(app)

    """Gracefully stop the paramter server on API shutdown."""
    logger.info("🛑 Gracefully stop the parameter server ...")   
    await ps.stop()


@app.get("/")
def root():
    return {"message": f"{settings.PROJECT_NAME} is running"}



@app.get("/status", response_model=Dict)
async def get_status():
    """Return the status of the Parameter Server."""
    return {"status": "running", "workers": list(ps.workers.keys())}


@app.get("/workers", response_model=Dict)
async def get_workers():
    """List all registered workers."""
    return {"workers": ps.workers}


@app.get("/weights", response_model=Dict)
async def get_global_weights():
    """Return the current global weights."""
    return {"global_weights": ps.global_weights}


@app.post("/broadcast-weights", response_model=Dict)
async def broadcast_weights():
    """
    Future: Trigger broadcast of global weights to all workers.
    (Placeholder for now)
    """
    return {"status": "not_implemented"}


@app.post("/train")
async def train_model(data: TrainRequest):
    """
    Submit training data to workers (placeholder: single worker simulation)
    """
    X = np.array(data.X)
    y = np.array(data.y)

    # For now, just store data in PS; workers will pick this up later
    ps.last_batch = {"X": X.tolist(), "y": y.tolist()}
    return {"status": "data_received", "shape": X.shape}




async def send_to_parameter_server(message: dict) -> dict:
    """
    Send a message to the Parameter Server over ZMQ and wait for response.
    """
    ctx = zmq.asyncio.Context.instance()
    socket = ctx.socket(zmq.REQ)
    socket.connect(settings.PARAMETER_SERVER_URL)  # PS is running locally
    try:
        await socket.send_json(message)
        reply = await socket.recv_json()
        return reply
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        socket.close()


@app.post("/training-jobs/start/{job_id}")
async def start_training_job(
    job_id: int = Path(..., description="Training job ID"),
    body: dict = Body(..., description="Payload from Django with init_params"),
):
    """
    Start a training job by sending request to Parameter Server.
    """
    print("---- JOB RECEIVED------------")
    try:
        init_params = body.get("init_params")
        if not init_params:
            raise HTTPException(status_code=400, detail="Missing init_params in payload")

        message = {
            "type": "create_job",
            "job_id": job_id,
            "init_params": init_params
        }

        response = await send_to_parameter_server(message)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))