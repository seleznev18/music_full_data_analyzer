"""
Caption generation via Gemini API (kie.ai proxy).

Sends an audio file (base64-encoded) and a text prompt to the kie.ai
Gemini chat-completions endpoint, returns the generated text.
"""

import base64
import json
import os
import subprocess
import tempfile
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

Write only the description paragraph, nothing else.

IMPORTANT: Output ONLY the music description as a single paragraph or two. \
Do not add any follow-up questions, offers to help, or conversational text. \
Do not write things like "Is there anything else I can help you with?", \
"Let me know if you need anything else", "Here's the description:", "Sure!", \
"Of course!", "Certainly!" etc. Start directly with the description and end \
immediately after it. No preamble, no sign-off.\
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

        # kie.ai only accepts mp3 — convert other formats to a temp mp3
        tmp_mp3: Path | None = None
        try:
            if audio_path.suffix.lower() != ".mp3":
                fd, tmp_name = tempfile.mkstemp(suffix=".mp3", prefix="caption_")
                os.close(fd)
                tmp_mp3 = Path(tmp_name)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(audio_path), "-q:a", "2", str(tmp_mp3)],
                    check=True,
                    capture_output=True,
                )
                send_path = tmp_mp3
            else:
                send_path = audio_path

            audio_bytes = send_path.read_bytes()
            b64_data = base64.b64encode(audio_bytes).decode("utf-8")
            data_uri = f"data:audio/mpeg;base64,{b64_data}"

            return self._call_api(data_uri)
        finally:
            if tmp_mp3 and tmp_mp3.exists():
                tmp_mp3.unlink()

    def _call_api(self, data_uri: str) -> str:
        """Send the base64 audio data URI to the kie.ai streaming endpoint."""
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
            "stream": True,
            "include_thoughts": False,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        content_parts: list[str] = []

        with httpx.Client(timeout=300.0) as client:
            with client.stream("POST", url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        content_parts.append(text)

        if not content_parts:
            raise RuntimeError("Gemini API returned no content")

        return "".join(content_parts)
