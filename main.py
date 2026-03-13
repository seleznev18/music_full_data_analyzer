"""
CLI entry point for music_full_data_analyzer.

Reads a CSV manifest from music_for_preprocessing/, and for each song:
  1. Extracts BPM, Key, Time Signature (Essentia)
  2. Fetches lyrics from Genius
  3. Generates a caption via Gemini API

Results are written to a JSON output file.

Usage:
    python main.py
    python main.py --input-dir ./music_for_preprocessing --output results.csv
    python main.py --skip-lyrics --skip-caption
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from config import settings
from src.audio_analysis.analyzers import (
    BpmAnalyzer,
    KeyAnalyzer,
    TimeSignatureAnalyzer,
)
from src.audio_analysis.service import AudioAnalysisService, AudioLoader
from src.caption.gemini_service import GeminiCaptionService
from src.lyrics.genius_provider import GeniusLyricsProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze music files: extract audio features, fetch lyrics, generate captions."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=settings.input_dir,
        help="Directory containing audio files and manifest.csv",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=settings.manifest_file,
        help="Name of the CSV manifest file inside input-dir",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=settings.output_file,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--skip-lyrics",
        action="store_true",
        help="Skip lyrics fetching from Genius",
    )
    parser.add_argument(
        "--skip-caption",
        action="store_true",
        help="Skip caption generation via Gemini",
    )
    return parser.parse_args()


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Read the CSV manifest (file_name, song_name, artist)."""
    if not manifest_path.exists():
        print(f"Error: manifest not found at {manifest_path}")
        sys.exit(1)

    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("Error: manifest is empty")
        sys.exit(1)

    required = {"file_name", "song_name", "artist"}
    if not required.issubset(rows[0].keys()):
        print(f"Error: manifest must have columns: {', '.join(required)}")
        sys.exit(1)

    return rows


def build_audio_service() -> AudioAnalysisService:
    loader = AudioLoader()
    analyzers = [KeyAnalyzer(), BpmAnalyzer(), TimeSignatureAnalyzer()]
    return AudioAnalysisService(loader, analyzers)


def build_lyrics_provider() -> GeniusLyricsProvider | None:
    if not settings.genius_api_token:
        return None
    return GeniusLyricsProvider(api_token=settings.genius_api_token)


def build_caption_service() -> GeminiCaptionService | None:
    if not settings.gemini_api_key:
        return None
    return GeminiCaptionService(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_model,
    )


def main() -> None:
    args = parse_args()

    input_dir: Path = args.input_dir
    manifest_path = input_dir / args.manifest
    entries = read_manifest(manifest_path)

    print(f"Loaded {len(entries)} songs from {manifest_path}")

    # --- Build services ---
    audio_service = build_audio_service()

    lyrics_provider = None
    if not args.skip_lyrics:
        lyrics_provider = build_lyrics_provider()
        if lyrics_provider is None:
            print("Warning: GENIUS_API_TOKEN not set, skipping lyrics")

    caption_service = None
    if not args.skip_caption:
        caption_service = build_caption_service()
        if caption_service is None:
            print("Warning: GEMINI_API_KEY not set, skipping captions")

    # --- Process each song ---
    results: list[dict[str, str]] = []

    for i, entry in enumerate(entries, 1):
        file_name = entry["file_name"]
        song_name = entry["song_name"]
        artist = entry["artist"]
        audio_path = input_dir / file_name

        print(f"\n[{i}/{len(entries)}] {song_name} - {artist} ({file_name})")

        row: dict[str, str] = {
            "file_name": file_name,
            "song_name": song_name,
            "artist": artist,
            "bpm": "",
            "key": "",
            "time_signature": "",
            "lyrics": "",
            "caption": "",
            "errors": "",
        }
        errors: list[str] = []

        # 1) Audio analysis
        if not audio_path.exists():
            errors.append(f"Audio file not found: {file_name}")
            print(f"  [SKIP] Audio file not found")
        else:
            print(f"  Analyzing audio...", end=" ", flush=True)
            features = audio_service.analyze_file(audio_path)
            if "error" in features:
                errors.append(f"Audio analysis: {features['error']}")
                print(f"ERROR - {features['error']}")
            else:
                row["bpm"] = str(features.get("bpm", ""))
                row["key"] = str(features.get("key", ""))
                row["time_signature"] = str(features.get("time_signature", ""))
                print(f"BPM={row['bpm']}, Key={row['key']}, TimeSignature={row['time_signature']}")

        # 2) Lyrics
        if lyrics_provider:
            print(f"  Fetching lyrics...", end=" ", flush=True)
            try:
                result = lyrics_provider.fetch_lyrics(song_name, artist)
                row["lyrics"] = result.lyrics
                preview = result.lyrics[:80].replace("\n", " ")
                print(f"OK ({len(result.lyrics)} chars) \"{preview}...\"")
            except Exception as exc:
                errors.append(f"Lyrics: {exc}")
                print(f"ERROR - {exc}")

        # 3) Caption
        if caption_service and audio_path.exists():
            print(f"  Generating caption...", end=" ", flush=True)
            try:
                caption = caption_service.generate_caption(audio_path)
                row["caption"] = caption
                preview = caption[:80].replace("\n", " ")
                print(f"OK \"{preview}...\"")
            except Exception as exc:
                errors.append(f"Caption: {exc}")
                print(f"ERROR - {exc}")

        row["errors"] = "; ".join(errors)
        results.append(row)

    # --- Write output ---
    output_path = Path(args.output)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Results written to {output_path}")
    successful = sum(1 for r in results if not r["errors"])
    print(f"  Total: {len(results)}, Successful: {successful}, With errors: {len(results) - successful}")


if __name__ == "__main__":
    main()
