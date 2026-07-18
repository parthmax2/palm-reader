import unittest

from app.generation.reader import _template_reading
from app.knowledge.engine import match_rules
from app.models.schemas import FeatureVector, LineFeature


class TemplateReaderTests(unittest.TestCase):
    def _sample_features(self):
        return FeatureVector(
            hand="right",
            heart=LineFeature(present=True, length="short", curved=False, point_count=41, confidence=0.68),
            head=LineFeature(present=True, length="long", curved=False, point_count=49, confidence=0.82),
            life=LineFeature(present=True, length="short", curved=True, point_count=55, confidence=0.92),
            fate=LineFeature(present=False, point_count=0, confidence=0.0),
            overall_confidence=0.81,
        )

    def test_fallback_fields_are_filled_from_detected_features(self):
        features = self._sample_features()

        reading = _template_reading(features, match_rules(features), "en")

        self.assertNotIn("No clear sign", reading.personality)
        self.assertIn("Fate line was not clearly detected", reading.career)
        self.assertIn("Most notable detected pattern", reading.signs)
        self.assertIn("confidence 0.82", reading.signs)
        self.assertIn("Fate line was not clearly detected", reading.signs)

    def test_hindi_fallback_uses_powerful_but_clear_voice(self):
        features = self._sample_features()

        reading = _template_reading(features, match_rules(features), "hi")
        combined = " ".join(reading.model_dump().values())

        self.assertIn("सामुद्रिक शास्त्र", reading.overview)
        self.assertIn("Mastishka Rekha", reading.personality)
        self.assertIn("निर्णय-शक्ति", reading.personality)
        self.assertIn("सरल शब्दों", reading.personality)
        self.assertIn("Bhagya Rekha", reading.career)
        self.assertIn("कर्म", reading.career)
        self.assertIn("Lakshan", reading.signs)
        self.assertNotIn("à", combined)


if __name__ == "__main__":
    unittest.main()
