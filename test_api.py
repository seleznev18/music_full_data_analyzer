"""Quick test of the kie.ai Gemini API connection."""

import json
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash")

if not API_KEY:
    print("ERROR: GEMINI_API_KEY not set in .env")
    exit(1)

url = f"https://api.kie.ai/{MODEL}/v1/chat/completions"

payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Say hello in one sentence."}
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

print(f"POST {url}")
print(f"Model: {MODEL}")
print(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
print("---")

with httpx.Client(timeout=60.0) as client:
    with client.stream("POST", url, json=payload, headers=headers) as resp:
        print(f"Status: {resp.status_code}")
        print(f"Headers: {dict(resp.headers)}")
        print("---RAW LINES---")
        for line in resp.iter_lines():
            print(repr(line))
        print("---END---")
