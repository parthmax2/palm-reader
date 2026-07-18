"""FastAPI application entry point.

Run:  uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Palmistry AI",
    description="Hindu palmistry (Samudrika Shastra) readings from a palm photo, in English and Hindi.",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(router, prefix="/v1")


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")
