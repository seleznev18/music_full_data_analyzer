"""
Batch processor: reads manifest.csv and sends each song to the analysis API.

Run the API server first in one terminal:
    uvicorn main:app --host 0.0.0.0 --port 8000

Then run this script in a second terminal:
    python process_songs.py

Optional arguments:
    --host   API base URL (default: http://localhost:8000)
    --output Output JSON file (default: results.json)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import requests

MANIFEST_PATH = Path("music_for_preprocessing/manifest.csv")
MUSIC_DIR = Path("music_for_preprocessing")
API_ENDPOINT = "/api/v1/analyze"


def analyze_song(base_url: str, audio_path: Path, song_name: str, artist: str) -> dict:
    url = base_url.rstrip("/") + API_ENDPOINT
    with open(audio_path, "rb") as f:
        response = requests.post(
            url,
            files={"file": (audio_path.name, f)},
            data={"song_name": song_name, "artist": artist},
            timeout=600,  # caption generation can take up to 5 min per song
        )
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Batch analyze songs via the analysis API")
    parser.add_argument("--host", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", default="results.json", help="Output file for results")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest not found at {MANIFEST_PATH}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        print("No entries found in manifest.csv.")
        return

    print(f"Found {len(rows)} song(s). Sending requests to {args.host} ...")

    results = []
    for i, row in enumerate(rows, 1):
        file_name = row["file_name"]
        song_name = row["song_name"]
        artist = row["artist"]
        audio_path = MUSIC_DIR / file_name

        print(f"[{i}/{len(rows)}] {song_name} — {artist} ...", end=" ", flush=True)

        if not audio_path.exists():
            print(f"SKIP (file not found: {audio_path})")
            results.append({"file_name": file_name, "error": "audio file not found"})
            continue

        try:
            result = analyze_song(args.host, audio_path, song_name, artist)
            print("OK")
            results.append(result)
        except requests.HTTPError as exc:
            msg = f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"
            print(f"FAILED ({msg})")
            results.append({"file_name": file_name, "song_name": song_name, "artist": artist, "error": msg})
        except Exception as exc:
            print(f"FAILED ({exc})")
            results.append({"file_name": file_name, "song_name": song_name, "artist": artist, "error": str(exc)})

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    success = sum(1 for r in results if "error" not in r)
    print(f"\nDone: {success}/{len(rows)} succeeded. Results saved to {output_path}")


if __name__ == "__main__":
    main()
