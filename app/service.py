"""Orchestration: ties CV pipeline -> rule engine -> reading generation."""
from __future__ import annotations

import time
import uuid

from app.config import settings
from app.generation.reader import generate_readings
from app.knowledge.engine import match_rules
from app.models.schemas import FeatureVector, ReadingOptions, ReadingResponse


def _build_response(fv: FeatureVector, options: ReadingOptions, started: float) -> ReadingResponse:
    rules = match_rules(fv)
    readings, mode = generate_readings(
        fv, rules, options.languages, options.detail, options.model
    )
    return ReadingResponse(
        reading_id=f"rd_{uuid.uuid4().hex[:12]}",
        hand=fv.hand,
        features=fv,
        matched_rules=[r.id for r in rules],
        readings=readings,
        disclaimer={"en": settings.disclaimer_en, "hi": settings.disclaimer_hi},
        overall_confidence=fv.overall_confidence,
        generation=mode,
        processing_ms=int((time.time() - started) * 1000),
    )


def reading_from_features(fv: FeatureVector, options: ReadingOptions) -> ReadingResponse:
    return _build_response(fv, options, time.time())


def reading_from_image(image_path: str, options: ReadingOptions) -> ReadingResponse:
    # Imported lazily so the API can start even if heavy CV deps are missing.
    from app.cv.pipeline import extract_features

    started = time.time()
    fv = extract_features(image_path)
    return _build_response(fv, options, started)
