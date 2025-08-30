"""
Config & settings loader for deep-distribute-cluster.

- Loads from environment variables and an optional .env file
- Central place for runtime role selection (ps|worker)
- Validates ports/URLs and training knobs
- Exposes a single `settings` object for imports

This file is intentionally framework-agnostic (no FastAPI imports).
"""
from __future__ import annotations

import ipaddress
import os
from typing import List, Optional, Tuple

from pydantic import  Field, validator, AnyHttpUrl, AnyUrl
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env.

    These settings are shared by both roles (Parameter Server and Worker).
    """

    # ----------------------------
    # Role & basic runtime
    # ----------------------------
    ROLE: str = Field("ps", description="Role for this process: 'ps' or 'worker'.")

    # Bindings for Parameter Server (HTTP & WS)
    PS_BIND_HOST: str = Field("0.0.0.0", description="PS HTTP/WS bind host")
    PS_HTTP_PORT: int = Field(8080, description="PS HTTP port for REST API")
    PS_WS_PORT: int = Field(8090, description="PS WebSocket port for streaming")

    # Public URLs for Workers to reach PS (required in worker mode)
    PS_PUBLIC_URL: Optional[AnyHttpUrl] = Field(
        None, description="Public/base HTTP URL for PS, e.g., http://192.168.1.10:8080"
    )
    PS_WS_URL: Optional[AnyUrl] = Field(
        None, description="Public/base WS URL for PS, e.g., ws://192.168.1.10:8090/ws"
    )

    # ----------------------------
    # PostgreSQL (shared with Django project)
    # ----------------------------
    PG_HOST: str = Field("localhost")
    PG_PORT: int = Field(5432)
    PG_DB: str = Field("deep-distribute")
    PG_USER: str = Field("postgres")
    PG_PASSWORD: str = Field("window")
    PG_CONN_MAX_AGE: int = Field(1, description="Seconds to keep pg connection alive")

    # ----------------------------
    # Dataset & IO
    # ----------------------------
    DATA_CACHE_DIR: str = Field("/tmp/deepdist-cache", description="Local cache for downloaded dataset files")
    DOWNLOAD_CONCURRENCY: int = Field(8, ge=1, le=64)

    # ----------------------------
    # Training defaults (can be overridden per job)
    # ----------------------------
    ALGO_NAME: str = Field("ResNet50")
    LEARNING_RATE: float = Field(5e-4)
    OPTIMIZER: str = Field("adam")
    LOSS: str = Field("binary_crossentropy")
    IMG_SIZE: str = Field("224,224", description="HxW or W,H accepted; parsed at runtime")
    EPOCHS: int = Field(10, ge=1)
    VALIDATION_SPLIT: float = Field(0.2, ge=0.0, le=0.9)
    RANDOM_SEED: int = Field(42)

    # Micro-step strategy for async updates
    MICRO_BATCH: int = Field(2, ge=1, le=16, description="Number of samples per micro-step (1–3 recommended)")
    AGGREGATION: str = Field(
        "asgd_ssp", description="Aggregator: asgd_ssp | fedbuff | topk"
    )
    TOPK_PCT: float = Field(1.0, ge=0.1, le=100.0, description="Top-k percentage for sparsification (if enabled)")
    QUANTIZATION: str = Field("fp16", description="fp32 | fp16 | q8")
    SSP_STALENESS: int = Field(2, ge=0, le=16, description="Max staleness window τ for SSP")
    QUORUM_FRACTION: float = Field(0.6, ge=0.1, le=1.0, description="K-of-N quorum fraction for aggregation")
    ROUND_TIMEOUT_MS: int = Field(100, ge=10, le=5000, description="Micro-round timeout in ms")

    # ----------------------------
    # Telemetry & logging
    # ----------------------------
    LOG_LEVEL: str = Field("INFO")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    # ----------------------------
    # Validators
    # ----------------------------
    @validator("ROLE")
    def validate_role(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in {"ps", "worker"}:
            raise ValueError("ROLE must be 'ps' or 'worker'")
        return v

    @validator("PS_HTTP_PORT", "PS_WS_PORT", pre=True)
    def validate_port(cls, v: int) -> int:
        if not (1 <= int(v) <= 65535):
            raise ValueError("Invalid TCP port")
        return int(v)

    @validator("PS_BIND_HOST")
    def validate_host(cls, v: str) -> str:
        try:
            if v != "0.0.0.0":
                ipaddress.ip_address(v)
        except Exception:
            # allow hostnames as well
            if not v or any(c.isspace() for c in v):
                raise ValueError("PS_BIND_HOST must be an IP or hostname")
        return v

    @validator("IMG_SIZE")
    def validate_img_size(cls, v: str) -> str:
        v = v.replace(" ", "")
        if "," not in v:
            raise ValueError("IMG_SIZE must be like '224,224'")
        a, b = v.split(",", 1)
        try:
            ah, aw = int(a), int(b)
        except Exception as e:
            raise ValueError("IMG_SIZE must be two integers separated by comma")
        if ah < 16 or aw < 16:
            raise ValueError("IMG_SIZE too small")
        return f"{ah},{aw}"

    @property
    def img_size_tuple(self) -> Tuple[int, int]:
        h, w = self.IMG_SIZE.split(",")
        return int(h), int(w)

    @property
    def is_ps(self) -> bool:
        return self.ROLE == "ps"

    @property
    def is_worker(self) -> bool:
        return self.ROLE == "worker"


# Singleton settings instance used across the app
settings = Settings()  # type: ignore


def dump_effective_config(mask_secrets: bool = True) -> dict:
    """Return a sanitized view of the current settings for diagnostics."""
    public = settings.dict()
    if mask_secrets:
        if "PG_PASSWORD" in public and public["PG_PASSWORD"]:
            public["PG_PASSWORD"] = "***"
    # Derived fields
    public["IMG_SIZE_TUPLE"] = settings.img_size_tuple
    public["ROLE_IS_PS"] = settings.is_ps
    public["ROLE_IS_WORKER"] = settings.is_worker
    return public
