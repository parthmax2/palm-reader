import unittest
from pathlib import Path

from app.main import root


class LandingPageTests(unittest.TestCase):
    def test_root_serves_palmistry_ai_landing_page(self):
        response = root()
        html = Path(response.path).read_text(encoding="utf-8")

        self.assertEqual(Path(response.path).name, "index.html")
        self.assertIn("Palmistry AI", html)
        self.assertIn("The Story Hidden in Your Palm", html)
        self.assertIn("Begin Palm Reading", html)


if __name__ == "__main__":
    unittest.main()
