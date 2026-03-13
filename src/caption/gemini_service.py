"""
Caption generation via Gemini API (kie.ai proxy).

Sends an audio file (base64-encoded) and a text prompt to the kie.ai
Gemini chat-completions endpoint, returns the generated text.
"""

import base64
from pathlib import Path

import httpx

CAPTION_PROMPT = """\
Analyze this music track and write a detailed production description that captures \
its sonic characteristics. The description must be a single paragraph covering:
- Genre and subgenre
- Vocal type and style (gender, tone, technique)
- Key instruments and sounds (synths, guitars, drums, bass, etc.)
- Notable production techniques (reverb, compression, effects, arrangement choices)
- Overall mood, atmosphere, and era/aesthetic

Here are examples of the expected output format:

Example 1: "Energetic J-pop track with female vocal, bright synthesizer leads, \
punchy electronic drums, and a driving bass line. The production features heavy \
reverb on vocals, sidechain compression on pads, and a drop section with layered \
harmonies. Uplifting, youthful atmosphere with a nostalgic 2010s anime soundtrack feel."

Example 2: "Gritty Southern hip-hop track with aggressive male vocal, deep 808 sub-bass, \
crisp clap-snare layering, and a looping minor-key piano motif. The mix features \
distorted ad-libs panned wide, tape-saturated master bus, and a half-time breakdown \
with pitched vocal chops. Dark, confrontational energy with a late-2010s trap aesthetic."

Example 3: "Lush indie folk ballad with breathy female vocal, fingerpicked nylon-string \
guitar, understated upright bass, and gentle brush-played drums. The production features \
warm analog saturation, close-mic room ambience, and a gradual string swell in the final \
chorus. Melancholic, introspective atmosphere with an early-2020s singer-songwriter feel."

Write only the description paragraph, nothing else.\
"""

KIE_BASE_URL = "https://api.kie.ai"


class GeminiCaptionService:
    """Generates text captions for audio files using Gemini via kie.ai."""

    def __init__(self, api_key: str, model_name: str = "gemini-3-flash") -> None:
        if not api_key:
            raise RuntimeError(
                "Gemini API key not configured. Set GEMINI_API_KEY in .env"
            )
        self._api_key = api_key
        self._model_name = model_name

    def generate_caption(self, audio_path: Path) -> str:
        """
        Encode an audio file as base64, send it to the kie.ai Gemini
        chat-completions endpoint, and return the generated caption.

        Parameters
        ----------
        audio_path:
            Path to the audio file on disk.

        Returns
        -------
        Generated text from Gemini.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        mime_map = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".m4a": "audio/mp4",
            ".aac": "audio/aac",
        }
        mime_type = mime_map.get(audio_path.suffix.lower(), "application/octet-stream")

        audio_bytes = audio_path.read_bytes()
        b64_data = base64.b64encode(audio_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{b64_data}"

        url = f"{KIE_BASE_URL}/{self._model_name}/v1/chat/completions"

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CAPTION_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "stream": False,
            "include_thoughts": False,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        with httpx.Client(timeout=300.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]
