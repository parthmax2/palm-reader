"""Reading generation via Gemini 2.5 Flash, with a template fallback when no key is set."""
from __future__ import annotations

import json

from app.config import settings
from app.models.schemas import FeatureVector, LocalizedReading, MatchedRule
from .prompts import RESPONSE_SCHEMA, SYSTEM_INSTRUCTION, build_user_prompt

# Which rule domains feed which reading section (used by the template fallback).
_SECTION_DOMAINS = {
    "personality": {"personality"},
    "relationships": {"relationships"},
    "career": {"career", "intellect"},
    "health": {"health"},
}

_PERSONALITY_FALLBACK_DOMAINS = {"intellect", "health", "relationships"}

_LINE_LABELS = {
    "en": {
        "heart": "Heart line",
        "head": "Head line",
        "life": "Life line",
        "fate": "Fate line",
    },
    "hi": {
        "heart": "\u0939\u0943\u0926\u092f \u0930\u0947\u0916\u093e",
        "head": "\u092e\u0938\u094d\u0924\u093f\u0937\u094d\u0915 \u0930\u0947\u0916\u093e",
        "life": "\u091c\u0940\u0935\u0928 \u0930\u0947\u0916\u093e",
        "fate": "\u092d\u093e\u0917\u094d\u092f \u0930\u0947\u0916\u093e",
    },
}

_FATE_NOT_DETECTED = {
    "en": (
        "The Fate line was not clearly detected, so career and direction are read mainly "
        "from the Head line; traditionally this can suggest a self-directed path shaped "
        "more by choices than fixed circumstance."
    ),
    "hi": (
        "\u092d\u093e\u0917\u094d\u092f \u0930\u0947\u0916\u093e \u0938\u094d\u092a\u0937\u094d\u091f "
        "\u0930\u0942\u092a \u0938\u0947 \u0928\u0939\u0940\u0902 \u0926\u093f\u0916\u0940, "
        "\u0907\u0938\u0932\u093f\u090f \u0915\u0930\u093f\u092f\u0930 \u0914\u0930 "
        "\u0926\u093f\u0936\u093e \u0915\u094b \u092e\u0941\u0916\u094d\u092f \u0930\u0942\u092a "
        "\u0938\u0947 \u092e\u0938\u094d\u0924\u093f\u0937\u094d\u0915 \u0930\u0947\u0916\u093e "
        "\u0938\u0947 \u092a\u0922\u093c\u093e \u0917\u092f\u093e \u0939\u0948; "
        "\u092a\u0930\u0902\u092a\u0930\u093e\u0917\u0924 \u0930\u0942\u092a \u0938\u0947 "
        "\u092f\u0939 \u091a\u092f\u0928\u094b\u0902 \u0914\u0930 \u092a\u094d\u0930\u092f\u093e\u0938 "
        "\u0938\u0947 \u092c\u0928\u0928\u0947 \u0935\u093e\u0932\u0947 \u0938\u094d\u0935\u0928\u093f\u0930\u094d\u092e\u093f\u0924 "
        "\u092e\u093e\u0930\u094d\u0917 \u0915\u093e \u0938\u0902\u0915\u0947\u0924 \u092e\u093e\u0928\u093e "
        "\u091c\u093e\u0924\u093e \u0939\u0948."
    ),
}

_HI_LENGTH = {"long": "\u0932\u0902\u092c\u0940", "short": "\u091b\u094b\u091f\u0940"}
_HI_CURVE = {True: "\u0918\u0941\u092e\u093e\u0935\u0926\u093e\u0930", False: "\u0938\u0940\u0927\u0940"}


def _combine(*parts: str) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


def _is_foundation_rule(rule: MatchedRule) -> bool:
    return rule.id.endswith("_present")


def _line_phrase(fv: FeatureVector, name: str, language: str) -> str:
    line = getattr(fv, name)
    label = _LINE_LABELS.get(language, _LINE_LABELS["en"])[name]
    if not line.present:
        if name != "fate":
            return ""
        return (
            f"{label} was not clearly detected."
            if language == "en"
            else f"{label} \u0938\u094d\u092a\u0937\u094d\u091f \u0930\u0942\u092a \u0938\u0947 \u0928\u0939\u0940\u0902 \u0926\u093f\u0916\u0940."
        )

    traits: list[str] = []
    if line.length is not None:
        traits.append(line.length.value if language == "en" else ("\u0932\u0902\u092c\u0940" if line.length.value == "long" else "\u091b\u094b\u091f\u0940"))
    if line.curved is not None:
        if language == "en":
            traits.append("curved" if line.curved else "straight")
        else:
            traits.append("\u0918\u0941\u092e\u093e\u0935\u0926\u093e\u0930" if line.curved else "\u0938\u0940\u0927\u0940")
    if line.fork_end:
        traits.append("forked" if language == "en" else "\u0926\u094d\u0935\u093f\u0936\u093e\u0916\u093e")

    if language == "en":
        description = " and ".join(traits) if traits else "present"
        return f"{label} appears {description} (confidence {line.confidence:.2f})."

    description = " \u0914\u0930 ".join(traits) if traits else "\u092e\u094c\u091c\u0942\u0926"
    return f"{label} {description} \u0926\u093f\u0916\u0924\u0940 \u0939\u0948 (\u0935\u093f\u0936\u094d\u0935\u093e\u0938 {line.confidence:.2f})."


def _signs_summary(fv: FeatureVector, language: str) -> str:
    phrases = [
        _line_phrase(fv, "head", language),
        _line_phrase(fv, "heart", language),
        _line_phrase(fv, "life", language),
        _line_phrase(fv, "fate", language),
    ]
    body = " ".join(p for p in phrases if p)
    if language == "hi":
        return f"\u092e\u0941\u0916\u094d\u092f \u0926\u093f\u0916\u0947 \u0938\u0902\u0915\u0947\u0924: {body}"
    return f"Most notable detected pattern: {body}"


def _hi_line_description(fv: FeatureVector, name: str) -> str:
    line = getattr(fv, name)
    label = _LINE_LABELS["hi"][name]
    if not line.present:
        return f"{label} \u0938\u094d\u092a\u0937\u094d\u091f \u0930\u0942\u092a \u0938\u0947 \u0928\u0939\u0940\u0902 \u0926\u093f\u0916\u0940"

    traits: list[str] = []
    if line.length is not None:
        traits.append(_HI_LENGTH[line.length.value])
    if line.curved is not None:
        traits.append(_HI_CURVE[line.curved])
    if line.fork_end:
        traits.append("\u0926\u094d\u0935\u093f\u0936\u093e\u0916\u093e")
    description = " \u0914\u0930 ".join(traits) if traits else "\u092e\u094c\u091c\u0942\u0926"
    return f"{label} {description} \u0926\u093f\u0916\u0924\u0940 \u0939\u0948"


def _hindi_template_reading(fv: FeatureVector) -> LocalizedReading:
    """Hindi fallback with a Sanskrit-inspired but plain-spoken reading voice."""
    overview = (
        "\u0938\u093e\u092e\u0941\u0926\u094d\u0930\u093f\u0915 \u0936\u093e\u0938\u094d\u0924\u094d\u0930 "
        "\u0915\u0940 \u0926\u0943\u0937\u094d\u091f\u093f \u0938\u0947 \u092f\u0939 \u0939\u0938\u094d\u0924\u0930\u0947\u0916\u093e "
        "\u092a\u093e\u0920 \u0906\u092a\u0915\u0940 \u0939\u0925\u0947\u0932\u0940 \u092e\u0947\u0902 "
        "\u0926\u093f\u0916\u0947 \u092a\u094d\u0930\u092e\u0941\u0916 \u0932\u0915\u094d\u0937\u0923\u094b\u0902 "
        "\u092a\u0930 \u0906\u0927\u093e\u0930\u093f\u0924 \u0939\u0948\u0964 \u092f\u0939 "
        "\u092d\u0935\u093f\u0937\u094d\u092f \u0915\u093e \u0926\u093e\u0935\u093e \u0928\u0939\u0940\u0902 "
        "\u0915\u0930\u0924\u093e; \u092f\u0939 \u0906\u092a\u0915\u0940 \u0930\u0947\u0916\u093e\u0913\u0902 "
        "\u0915\u094b \u092a\u0930\u0902\u092a\u0930\u093e\u0917\u0924 \u0939\u0938\u094d\u0924\u0930\u0947\u0916\u093e "
        "\u0915\u0940 \u092d\u093e\u0937\u093e \u092e\u0947\u0902 \u0938\u092e\u091d\u093e\u0924\u093e \u0939\u0948\u0964"
    )

    if fv.head.present:
        personality = (
            f"{_hi_line_description(fv, 'head')}\u0964 "
            "\u0938\u093e\u092e\u0941\u0926\u094d\u0930\u093f\u0915 \u0936\u093e\u0938\u094d\u0924\u094d\u0930 "
            "\u092e\u0947\u0902 \u092e\u0938\u094d\u0924\u093f\u0937\u094d\u0915 \u0930\u0947\u0916\u093e "
            "(Mastishka Rekha) \u0915\u094b \u0935\u093f\u0935\u0947\u0915, \u0935\u093f\u091a\u093e\u0930-\u0936\u0915\u094d\u0924\u093f "
            "\u0914\u0930 \u0928\u093f\u0930\u094d\u0923\u092f-\u0936\u0915\u094d\u0924\u093f \u0915\u093e "
            "\u0932\u0915\u094d\u0937\u0923 \u092e\u093e\u0928\u093e \u091c\u093e\u0924\u093e \u0939\u0948\u0964 "
            "\u0938\u0930\u0932 \u0936\u092c\u094d\u0926\u094b\u0902 \u092e\u0947\u0902, \u0906\u092a "
            "\u091a\u0940\u091c\u094b\u0902 \u0915\u094b \u0938\u0924\u0939 \u092a\u0930 \u0928\u0939\u0940\u0902 "
            "\u091b\u094b\u0921\u093c\u0924\u0947; \u092a\u0939\u0932\u0947 \u0938\u092e\u091d\u0924\u0947 "
            "\u0939\u0948\u0902, \u092b\u093f\u0930 \u0920\u094b\u0938 \u0928\u093f\u0930\u094d\u0923\u092f "
            "\u0932\u0947\u0924\u0947 \u0939\u0948\u0902\u0964"
        )
    else:
        personality = (
            "\u092e\u0938\u094d\u0924\u093f\u0937\u094d\u0915 \u0930\u0947\u0916\u093e "
            "\u0938\u094d\u092a\u0937\u094d\u091f \u0928\u0939\u0940\u0902 \u0926\u093f\u0916\u0940, "
            "\u0907\u0938\u0932\u093f\u090f \u0935\u094d\u092f\u0915\u094d\u0924\u093f\u0924\u094d\u0935 "
            "\u0915\u094b \u0939\u0932\u094d\u0915\u0947 \u0938\u0902\u0915\u0947\u0924\u094b\u0902 "
            "\u0915\u0947 \u0938\u093e\u0925 \u0939\u0940 \u092a\u0922\u093c\u093e \u0917\u092f\u093e \u0939\u0948\u0964"
        )

    if fv.heart.present:
        relationships = (
            f"{_hi_line_description(fv, 'heart')}\u0964 "
            "\u0939\u0943\u0926\u092f \u0930\u0947\u0916\u093e (Hridaya Rekha) "
            "\u092d\u093e\u0935\u0928\u093e, \u092a\u094d\u0930\u0947\u092e \u0914\u0930 "
            "\u0930\u093f\u0936\u094d\u0924\u094b\u0902 \u092e\u0947\u0902 \u0938\u0902\u092f\u092e "
            "\u0915\u093e \u092a\u094d\u0930\u092e\u0941\u0916 \u0932\u0915\u094d\u0937\u0923 "
            "\u092e\u093e\u0928\u0940 \u091c\u093e\u0924\u0940 \u0939\u0948\u0964 "
            "\u0938\u0930\u0932 \u092d\u093e\u0937\u093e \u092e\u0947\u0902, \u0906\u092a "
            "\u0930\u093f\u0936\u094d\u0924\u094b\u0902 \u092e\u0947\u0902 \u0908\u092e\u093e\u0928\u0926\u093e\u0930\u0940, "
            "\u0938\u094d\u092a\u0937\u094d\u091f\u0924\u093e \u0914\u0930 \u0905\u092a\u0928\u0947 "
            "\u0928\u093f\u091c\u0940 \u0938\u094d\u0925\u093e\u0928 \u0915\u094b \u092e\u0939\u0924\u094d\u0935 "
            "\u0926\u0947\u0924\u0947 \u0939\u0948\u0902\u0964"
        )
    else:
        relationships = "\u0939\u0943\u0926\u092f \u0930\u0947\u0916\u093e \u0938\u094d\u092a\u0937\u094d\u091f \u0928\u0939\u0940\u0902 \u0926\u093f\u0916\u0940, \u0907\u0938\u0932\u093f\u090f \u092a\u094d\u0930\u0947\u092e \u0914\u0930 \u0930\u093f\u0936\u094d\u0924\u094b\u0902 \u0915\u0947 \u0938\u0902\u0915\u0947\u0924 \u0938\u0940\u092e\u093f\u0924 \u0930\u0916\u0947 \u0917\u090f \u0939\u0948\u0902\u0964"

    if fv.fate.present:
        career_opening = f"{_hi_line_description(fv, 'fate')}\u0964 "
    else:
        career_opening = (
            "\u092d\u093e\u0917\u094d\u092f \u0930\u0947\u0916\u093e (Bhagya Rekha) "
            "\u0938\u094d\u092a\u0937\u094d\u091f \u0930\u0942\u092a \u0938\u0947 \u0928\u0939\u0940\u0902 "
            "\u0926\u093f\u0916\u0940\u0964 "
        )
    career = (
        career_opening
        + "\u092a\u0930\u0902\u092a\u0930\u093e \u092e\u0947\u0902 \u092d\u093e\u0917\u094d\u092f "
        "\u0930\u0947\u0916\u093e \u0915\u094b \u0915\u0930\u094d\u092e, \u0926\u093f\u0936\u093e "
        "\u0914\u0930 \u091c\u0940\u0935\u0928-\u092a\u0925 \u0915\u0947 \u0938\u093e\u0925 "
        "\u091c\u094b\u0921\u093c\u093e \u091c\u093e\u0924\u093e \u0939\u0948\u0964 "
        "\u0906\u092a\u0915\u0947 \u092a\u093e\u0920 \u092e\u0947\u0902 \u0915\u0930\u093f\u092f\u0930 "
        "\u0915\u0940 \u0926\u093f\u0936\u093e \u0915\u094b \u092e\u0938\u094d\u0924\u093f\u0937\u094d\u0915 "
        "\u0930\u0947\u0916\u093e \u0915\u0947 \u0935\u093f\u0935\u0947\u0915 \u0914\u0930 "
        "\u0928\u093f\u0930\u094d\u0923\u092f-\u0936\u0915\u094d\u0924\u093f \u0938\u0947 \u092d\u0940 "
        "\u092a\u0922\u093c\u093e \u0917\u092f\u093e \u0939\u0948; \u0905\u0930\u094d\u0925\u093e\u0924 "
        "\u0906\u092a\u0915\u093e \u092e\u093e\u0930\u094d\u0917 \u092d\u093e\u0917\u094d\u092f "
        "\u0938\u0947 \u091c\u094d\u092f\u093e\u0926\u093e \u092f\u094b\u091c\u0928\u093e, "
        "\u092e\u0947\u0939\u0928\u0924 \u0914\u0930 \u0938\u091a\u0947\u0924 \u091a\u092f\u0928\u094b\u0902 "
        "\u0938\u0947 \u092c\u0928\u0924\u093e \u0926\u093f\u0916\u0924\u093e \u0939\u0948\u0964"
    )

    if fv.life.present:
        health = (
            f"{_hi_line_description(fv, 'life')}\u0964 "
            "\u091c\u0940\u0935\u0928 \u0930\u0947\u0916\u093e (Jeevan Rekha) "
            "\u0915\u094b \u091c\u0940\u0935\u0928-\u0936\u0915\u094d\u0924\u093f, "
            "\u090a\u0930\u094d\u091c\u093e \u0914\u0930 \u0906\u0924\u094d\u092e\u092c\u0932 "
            "\u0915\u093e \u0932\u0915\u094d\u0937\u0923 \u092e\u093e\u0928\u093e \u091c\u093e\u0924\u093e "
            "\u0939\u0948\u0964 \u092f\u0939 \u0906\u092f\u0941 \u0915\u0940 \u092d\u0935\u093f\u0937\u094d\u092f\u0935\u093e\u0923\u0940 "
            "\u0928\u0939\u0940\u0902 \u0915\u0930\u0924\u0940; \u092f\u0939 \u0906\u092a\u0915\u0947 "
            "\u0938\u094d\u0935\u092d\u093e\u0935, \u0909\u0924\u094d\u0938\u093e\u0939 \u0914\u0930 "
            "\u091c\u0940\u0935\u0928 \u091c\u0940\u0928\u0947 \u0915\u0940 \u0936\u0948\u0932\u0940 "
            "\u0915\u094b \u0938\u092e\u091d\u093e\u0924\u0940 \u0939\u0948\u0964"
        )
    else:
        health = "\u091c\u0940\u0935\u0928 \u0930\u0947\u0916\u093e \u0938\u094d\u092a\u0937\u094d\u091f \u0928\u0939\u0940\u0902 \u0926\u093f\u0916\u0940, \u0907\u0938\u0932\u093f\u090f \u090a\u0930\u094d\u091c\u093e \u0914\u0930 \u091c\u0940\u0935\u0928-\u0936\u0915\u094d\u0924\u093f \u0915\u0947 \u0938\u0902\u0915\u0947\u0924 \u0938\u0940\u092e\u093f\u0924 \u0930\u0916\u0947 \u0917\u090f \u0939\u0948\u0902\u0964"

    signs = (
        "\u092e\u0941\u0916\u094d\u092f \u0932\u0915\u094d\u0937\u0923 (Lakshan): "
        f"{_hi_line_description(fv, 'head')}, "
        f"{_hi_line_description(fv, 'heart')}, "
        f"{_hi_line_description(fv, 'life')}, \u0914\u0930 "
        f"{_hi_line_description(fv, 'fate')}\u0964 "
        "\u0915\u0941\u0932 \u092e\u093f\u0932\u093e\u0915\u0930 \u092f\u0939 "
        "\u0935\u093f\u0935\u0947\u0915, \u0938\u0902\u092f\u092e, \u0906\u0924\u094d\u092e\u092c\u0932 "
        "\u0914\u0930 \u0915\u0930\u094d\u092e-\u092a\u094d\u0930\u0927\u093e\u0928 "
        "\u091c\u0940\u0935\u0928-\u0926\u0943\u0937\u094d\u091f\u093f \u0915\u0940 "
        "\u0913\u0930 \u0938\u0902\u0915\u0947\u0924 \u0915\u0930\u0924\u093e \u0939\u0948\u0964"
    )

    return LocalizedReading(
        overview=overview,
        personality=personality,
        relationships=relationships,
        career=career,
        health=health,
        signs=signs,
    )


def _gemini_reading(fv: FeatureVector, rules: list[MatchedRule], language: str, detail: str,
                    model: str) -> LocalizedReading:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)
    prompt = build_user_prompt(fv, rules, language, detail)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=settings.gemini_temperature,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            thinking_config=types.ThinkingConfig(
                thinking_budget=settings.gemini_thinking_budget
            ),
        ),
    )
    data = json.loads(response.text)
    return LocalizedReading(**data)


def _template_reading(fv: FeatureVector, rules: list[MatchedRule], language: str) -> LocalizedReading:
    """Deterministic fallback: compose sections directly from matched rule meanings."""
    if language == "hi":
        return _hindi_template_reading(fv)

    attr = "meaning_hi" if language == "hi" else "meaning_en"

    def join(domains: set[str], *, limit: int | None = None,
             include_foundation: bool = True) -> str:
        picked = [
            getattr(r, attr)
            for r in rules
            if r.domain in domains and (include_foundation or not _is_foundation_rule(r))
        ]
        if limit is not None:
            picked = picked[:limit]
        return " ".join(picked)

    if language == "hi":
        overview = "आपकी हथेली की प्रमुख रेखाओं के आधार पर यह एक पारंपरिक हस्तरेखा पाठ है।"
        default = "इस क्षेत्र के लिए कोई स्पष्ट संकेत नहीं मिला।"
    else:
        overview = "Based on the principal lines of your palm, here is a traditional reading."
        default = "No clear sign was detected for this area."

    personality = join(_SECTION_DOMAINS["personality"])
    if not personality:
        personality = join(
            _PERSONALITY_FALLBACK_DOMAINS,
            limit=3,
            include_foundation=False,
        )

    career = join(_SECTION_DOMAINS["career"])
    if not fv.fate.present:
        career = _combine(career, _FATE_NOT_DETECTED.get(language, _FATE_NOT_DETECTED["en"]))

    return LocalizedReading(
        overview=overview,
        personality=personality or default,
        relationships=join(_SECTION_DOMAINS["relationships"]) or default,
        career=career or default,
        health=join(_SECTION_DOMAINS["health"]) or default,
        signs=_signs_summary(fv, language) or default,
    )


def generate_readings(fv: FeatureVector, rules: list[MatchedRule], languages: list[str],
                      detail: str = "standard", model: str | None = None):
    """Return ({lang: LocalizedReading}, generation_mode)."""
    model = model or settings.gemini_model
    readings: dict[str, LocalizedReading] = {}

    if settings.gemini_enabled:
        try:
            for lang in languages:
                readings[lang] = _gemini_reading(fv, rules, lang, detail, model)
            return readings, "gemini"
        except Exception as exc:  # fall back rather than fail the request
            print(f"[generation] Gemini call failed, using template fallback: {exc}")

    for lang in languages:
        readings[lang] = _template_reading(fv, rules, lang)
    return readings, "template-fallback"
