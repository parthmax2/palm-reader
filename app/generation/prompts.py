"""Prompt construction for grounded, bilingual palm-reading generation."""
from __future__ import annotations

from app.models.schemas import FeatureVector, MatchedRule

SYSTEM_INSTRUCTION = """You are an experienced, warm Hindu palm reader (a practitioner of \
Samudrika Shastra / Hast Rekha Shastra). You give readings that feel personal, respectful, \
and encouraging.

STRICT RULES — you must follow all of them:
1. Narrate ONLY the interpretations provided to you in the "matched interpretations" list. \
Do NOT invent new predictions, signs, or lines that were not given.
2. Never make deterministic claims. Never predict lifespan, death, specific dates, exact sums \
of money, disease diagnoses, or guaranteed events. Frame everything as tendencies, inclinations, \
and traditional meaning.
3. The Life line reveals vitality and energy — it has NOTHING to do with how long a person lives. \
Never imply otherwise.
4. Be warm and specific, but grounded. Weave the given interpretations into flowing prose.
5. You produce the reading in the requested language with natural, fluent phrasing.
6. For Hindi, use a powerful but understandable Samudrika Shastra voice: include Sanskrit/Hindi \
terms such as Hridaya Rekha, Mastishka Rekha, Jeevan Rekha, Bhagya Rekha, vivek, karma, \
nirnay-shakti, jeevan-shakti, and lakshan, but immediately explain them in simple modern Hindi. \
Do not sound overly mysterious, fatalistic, or like a literal translation.

Return your answer as JSON only, matching the requested schema."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "overview": {"type": "string"},
        "personality": {"type": "string"},
        "relationships": {"type": "string"},
        "career": {"type": "string"},
        "health": {"type": "string"},
        "signs": {"type": "string"},
    },
    "required": ["overview", "personality", "relationships", "career", "health", "signs"],
}


def _features_summary(fv: FeatureVector) -> str:
    parts = [f"Hand analysed: {fv.hand}."]
    for name, label in (("heart", "Heart line (Hridaya Rekha)"),
                        ("head", "Head line (Mastishka Rekha)"),
                        ("life", "Life line (Jeevan Rekha)"),
                        ("fate", "Fate line (Bhagya Rekha)")):
        line = getattr(fv, name)
        if line.present:
            bits = []
            if line.length:
                bits.append(f"{line.length.value}")
            if line.curved is not None:
                bits.append("curved" if line.curved else "straight")
            parts.append(f"{label}: present ({', '.join(bits) or 'detected'}).")
        else:
            parts.append(f"{label}: not clearly detected.")
    return " ".join(parts)


def build_user_prompt(fv: FeatureVector, rules: list[MatchedRule], language: str, detail: str) -> str:
    lang_name = {"en": "English", "hi": "Hindi (हिंदी)"}.get(language, language)
    length_hint = {
        "short": "Keep each section to 1-2 sentences.",
        "standard": "Keep each section to 2-4 sentences.",
        "detailed": "Each section may be 4-6 sentences, richer and more descriptive.",
    }.get(detail, "Keep each section to 2-4 sentences.")

    meaning_attr = "meaning_hi" if language == "hi" else "meaning_en"
    interp_lines = "\n".join(
        f"- [{r.domain}] {getattr(r, meaning_attr)}" for r in rules
    ) or "- (No strong signs were detected; give a gentle, general reading.)"

    hindi_style = ""
    if language == "hi":
        hindi_style = """
Hindi voice requirements:
- Write in Devanagari Hindi.
- Tone: shaktishali, Sanskrit-inspired, premium, but clear for normal readers.
- Use the pattern: Sanskrit/Hindi line name -> what was detected -> traditional meaning -> simple explanation.
- Prefer words like सामुद्रिक शास्त्र, हस्तरेखा, विवेक, निर्णय-शक्ति, जीवन-शक्ति, कर्म, दिशा, संयम, आत्मबल, ऊर्जा, लक्षण.
- Avoid overusing heavy words without explanation. Avoid दैवीय, अलौकिक, निश्चित भविष्य, and guaranteed predictions.
- Example tone: "आपकी मस्तिष्क रेखा (Mastishka Rekha) लंबी और सीधी दिखती है। सामुद्रिक शास्त्र में इसे विवेक और निर्णय-शक्ति का लक्षण माना जाता है। सरल शब्दों में, आप पहले समझते हैं, फिर ठोस निर्णय लेते हैं."
"""

    return f"""Write a palm reading in {lang_name}.

Detected palm features:
{_features_summary(fv)}

Matched interpretations (narrate ONLY these; do not add new claims):
{interp_lines}

Produce JSON with these sections, all written in {lang_name}:
- overview: a warm 2-3 sentence opening.
- personality: character and temperament.
- relationships: love and emotional life (from the Heart line).
- career: work, ambition, and thinking style (from Head line and combinations).
- health: vitality and energy (from the Life line; never mention lifespan).
- signs: a short note on the most notable detected sign.

{length_hint}
{hindi_style}"""
