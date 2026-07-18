"""Application configuration, loaded from environment / .env file."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_thinking_budget: int = 0
    gemini_temperature: float = 0.6

    # Privacy
    delete_images_after_processing: bool = True

    # The disclaimer appended to every reading (biometric/legal safety).
    disclaimer_en: str = (
        "This reading is a traditional interpretation based on Hindu palmistry "
        "(Samudrika Shastra), offered for guidance and entertainment. It is not "
        "medical, financial, legal, or predictive advice."
    )
    disclaimer_hi: str = (
        "यह पाठ हिंदू हस्तरेखा शास्त्र (सामुद्रिक शास्त्र) पर आधारित एक पारंपरिक "
        "व्याख्या है, जो मार्गदर्शन और मनोरंजन के लिए प्रस्तुत की गई है। यह कोई "
        "चिकित्सा, वित्तीय, कानूनी या भविष्यसूचक सलाह नहीं है।"
    )

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key)


settings = Settings()
