"""API routes."""
from __future__ import annotations

import json
import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.models.schemas import FromFeaturesRequest, ReadingOptions, ReadingResponse
from app.service import reading_from_features, reading_from_image

router = APIRouter()

_ALLOWED = {".jpg", ".jpeg", ".png", ".heic"}


@router.get("/health")
def health():
    return {
        "status": "ok",
        "gemini_enabled": settings.gemini_enabled,
        "gemini_model": settings.gemini_model,
    }


@router.post("/readings/from-features", response_model=ReadingResponse)
def readings_from_features(req: FromFeaturesRequest):
    """Generate a bilingual reading from a manually-supplied feature vector (no image)."""
    return reading_from_features(req.features, req.options)


@router.post("/readings", response_model=ReadingResponse)
async def readings_from_image(
    image: UploadFile = File(...),
    options: str = Form(default="{}"),
):
    """Full pipeline: palm image -> features -> rules -> bilingual reading."""
    ext = os.path.splitext(image.filename or "")[1].lower()
    if ext not in _ALLOWED:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(_ALLOWED)}")

    try:
        opts = ReadingOptions(**json.loads(options)) if options else ReadingOptions()
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        raise HTTPException(400, f"Invalid options JSON: {e}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        tmp.write(await image.read())
        tmp.close()
        from app.cv.pipeline import PalmNotDetected
        try:
            return reading_from_image(tmp.name, opts)
        except PalmNotDetected as e:
            raise HTTPException(422, f"Palm not detected: {e}")
    finally:
        if settings.delete_images_after_processing and os.path.exists(tmp.name):
            os.remove(tmp.name)


@router.post("/features")
async def features_only(image: UploadFile = File(...)):
    """CV only: palm image -> feature vector (no reading generation)."""
    ext = os.path.splitext(image.filename or "")[1].lower()
    if ext not in _ALLOWED:
        raise HTTPException(400, f"Unsupported file type '{ext}'.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        tmp.write(await image.read())
        tmp.close()
        from app.cv.pipeline import extract_features, PalmNotDetected
        try:
            return extract_features(tmp.name)
        except PalmNotDetected as e:
            raise HTTPException(422, f"Palm not detected: {e}")
    finally:
        if settings.delete_images_after_processing and os.path.exists(tmp.name):
            os.remove(tmp.name)
