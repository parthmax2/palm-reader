"""Pydantic schemas: the contracts between CV pipeline, rule engine, and generation."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Feature vector (output of the CV pipeline, input to the rule engine)
# --------------------------------------------------------------------------- #
class LineLength(str, Enum):
    short = "short"
    long = "long"


class LineFeature(BaseModel):
    """A single detected palm line."""
    present: bool = False
    length: Optional[LineLength] = None
    curved: Optional[bool] = None          # curved vs straight (heuristic)
    fork_end: Optional[bool] = None        # branches/forks at the terminating end
    breaks: Optional[int] = None           # number of breaks (None = not yet assessed)
    island: Optional[bool] = None          # island formation (None = not yet assessed)
    point_count: int = 0                   # detected polyline length (extent proxy)
    confidence: float = 0.0                # 0-1 detection confidence


class FeatureVector(BaseModel):
    """Structured description of a palm. Sanskrit names noted for the knowledge base."""
    hand: str = "unknown"                  # left | right | unknown
    # Lines detected by the CV pipeline:
    heart: LineFeature = Field(default_factory=LineFeature)   # Hridaya Rekha
    head: LineFeature = Field(default_factory=LineFeature)    # Mastishka Rekha
    life: LineFeature = Field(default_factory=LineFeature)    # Jeevan Rekha
    fate: LineFeature = Field(default_factory=LineFeature)    # Bhagya Rekha (best-effort)
    # Reserved for later pipeline stages (mounts, hand shape):
    quality_flags: list[str] = Field(default_factory=list)
    overall_confidence: float = 0.0

    _LINES = ("heart", "head", "life", "fate")

    def as_conditions(self) -> dict:
        """Flatten to a dot-keyed dict the rule engine matches against."""
        cond: dict = {"hand": self.hand}
        for name in self._LINES:
            line: LineFeature = getattr(self, name)
            cond[f"{name}.present"] = line.present
            if line.length is not None:
                cond[f"{name}.length"] = line.length.value
            if line.curved is not None:
                cond[f"{name}.curved"] = line.curved
            if line.fork_end is not None:
                cond[f"{name}.fork_end"] = line.fork_end
            if line.breaks is not None:
                cond[f"{name}.breaks_present"] = line.breaks > 0
        return cond


# --------------------------------------------------------------------------- #
# Rule engine output
# --------------------------------------------------------------------------- #
class MatchedRule(BaseModel):
    id: str
    domain: str
    meaning_en: str
    meaning_hi: str
    source: str
    weight: float


# --------------------------------------------------------------------------- #
# Reading (final output)
# --------------------------------------------------------------------------- #
class LocalizedReading(BaseModel):
    overview: str = ""
    personality: str = ""
    relationships: str = ""
    career: str = ""
    health: str = ""
    signs: str = ""


class ReadingResponse(BaseModel):
    reading_id: str
    hand: str
    features: FeatureVector
    matched_rules: list[str]
    readings: dict[str, LocalizedReading]   # {"en": ..., "hi": ...}
    disclaimer: dict[str, str]              # {"en": ..., "hi": ...}
    overall_confidence: float
    generation: str                          # "gemini" | "template-fallback"
    processing_ms: int


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class ReadingOptions(BaseModel):
    languages: list[str] = ["en", "hi"]
    detail: str = "standard"                 # short | standard | detailed
    model: Optional[str] = None              # override Gemini model


class FromFeaturesRequest(BaseModel):
    """Generate a reading from a manually-supplied feature vector (no image / CV)."""
    features: FeatureVector
    options: ReadingOptions = Field(default_factory=ReadingOptions)
