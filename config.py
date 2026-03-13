from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    genius_api_token: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash"

    model_config = {"env_file": ".env"}


settings = Settings()
