"""
Caption generation via Google Gemini API.

Sends an audio file and a text prompt to Gemini, returns the generated text.
"""

from pathlib import Path

import google.generativeai as genai


class GeminiCaptionService:
    """Generates text captions for audio files using Gemini."""

    DEFAULT_PROMPT = "Describe this music track."

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash") -> None:
        if not api_key:
            raise RuntimeError(
                "Gemini API key not configured. Set GEMINI_API_KEY in .env"
            )
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)

    def generate_caption(
        self, audio_path: Path, prompt: str | None = None
    ) -> str:
        """
        Upload an audio file to Gemini and generate a text caption.

        Parameters
        ----------
        audio_path:
            Path to the audio file on disk.
        prompt:
            Text prompt to send alongside the audio. Falls back to DEFAULT_PROMPT.

        Returns
        -------
        Generated text from Gemini.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        prompt_text = prompt or self.DEFAULT_PROMPT
        audio_file = genai.upload_file(path=str(audio_path))

        response = self._model.generate_content([prompt_text, audio_file])
        return response.text
