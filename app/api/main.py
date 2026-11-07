"""FastAPI entry point: uvicorn app.api.main:app"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.dependencies import get_state
from app.api.routes import analyze, datasets, evaluation, health
from app.core.logging_setup import configure_logging


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    get_state()  # load agents, manifest and model once at startup
    yield


app = FastAPI(
    title="MultiFusionAgent API",
    version="0.1.0",
    description="Educational multimodal research prototype. Not a medical device; outputs are not diagnoses.",
    lifespan=lifespan,
)
app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(datasets.router)
app.include_router(evaluation.router)
