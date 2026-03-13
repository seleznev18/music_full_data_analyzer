from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    genius_api_token: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_prompt: str = "Describe this music track."

    input_dir: Path = Path("music_for_preprocessing")
    manifest_file: str = "manifest.csv"
    output_file: str = "results.json"

    model_config = {"env_file": ".env"}


settings = Settings()
