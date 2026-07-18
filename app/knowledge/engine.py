"""Rule engine: match a FeatureVector against the Samudrika Shastra rulebook."""
from __future__ import annotations

import json
import os

from app.models.schemas import FeatureVector, MatchedRule

_RULES_PATH = os.path.join(os.path.dirname(__file__), "rules.json")

with open(_RULES_PATH, encoding="utf-8") as f:
    _RULEBOOK = json.load(f)

RULEBOOK_VERSION = _RULEBOOK.get("version", "unknown")
_RULES = _RULEBOOK["rules"]


def _matches(conditions: dict, feature_conditions: dict) -> bool:
    """Every condition in the rule must equal the feature value (all-of semantics)."""
    for key, expected in conditions.items():
        if feature_conditions.get(key) != expected:
            return False
    return True


def match_rules(features: FeatureVector) -> list[MatchedRule]:
    """Return matched rules, highest weight first."""
    fc = features.as_conditions()
    matched: list[MatchedRule] = []
    for rule in _RULES:
        if _matches(rule["conditions"], fc):
            matched.append(
                MatchedRule(
                    id=rule["id"],
                    domain=rule["domain"],
                    meaning_en=rule["meaning_en"],
                    meaning_hi=rule["meaning_hi"],
                    source=rule["source"],
                    weight=rule["weight"],
                )
            )
    matched.sort(key=lambda r: r.weight, reverse=True)
    return matched
