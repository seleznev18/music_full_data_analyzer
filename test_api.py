"""Quick test of the kie.ai Gemini API with an audio file."""

import base64
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash")

if not API_KEY:
    print("ERROR: GEMINI_API_KEY not set in .env")
    exit(1)

# Find first audio file in music_for_preprocessing/
audio_dir = Path("music_for_preprocessing")
audio_file = None
for ext in ("*.flac", "*.mp3", "*.wav", "*.ogg", "*.m4a", "*.aac"):
    files = list(audio_dir.glob(ext))
    if files:
        audio_file = files[0]
        break

if not audio_file:
    print("ERROR: No audio file found in music_for_preprocessing/")
    exit(1)

mime_map = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
}
mime_type = mime_map.get(audio_file.suffix.lower(), "application/octet-stream")

print(f"Audio file: {audio_file} ({audio_file.stat().st_size / 1024 / 1024:.1f} MB)")
print(f"MIME type: {mime_type}")

audio_bytes = audio_file.read_bytes()
b64_data = base64.b64encode(audio_bytes).decode("utf-8")
data_uri = f"data:{mime_type};base64,{b64_data}"

print(f"Base64 payload size: {len(b64_data) / 1024 / 1024:.1f} MB")

url = f"https://api.kie.ai/{MODEL}/v1/chat/completions"

payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this audio in one sentence."},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }
    ],
    "stream": True,
    "include_thoughts": False,
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

print(f"\nPOST {url}")
print("Sending request (this may take a while for large files)...")
print("---RAW LINES---")

with httpx.Client(timeout=300.0) as client:
    with client.stream("POST", url, json=payload, headers=headers) as resp:
        print(f"Status: {resp.status_code}")
        for line in resp.iter_lines():
            if line:
                print(line)
        print("---END---")
