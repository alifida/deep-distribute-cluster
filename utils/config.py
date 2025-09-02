"""
Configuration utilities for the FastAPI app.

Uses environment variables with defaults.
"""

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Deep Distribute Cluster"
    VERSION: str = "1.0.0"

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./training.db")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Redis / Dispatcher
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    JOB_QUEUE_NAME: str = os.getenv("JOB_QUEUE_NAME", "deepdist:jobs:queue")
    JOB_CHANNEL: str = os.getenv("JOB_CHANNEL", "deepdist:jobs:available")
    JOB_CLAIM_TTL: int = int(os.getenv("JOB_CLAIM_TTL", "30"))  # seconds



    PARAMETER_SERVER_URL: str = os.getenv("PARAMETER_SERVER_URL", "tcp://192.168.10.120:5555")
    WORKER_HOST: str = os.getenv("WORKER_HOST", "locahost")

    class Config:
        env_file = ".env"


settings = Settings()
