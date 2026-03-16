"""
ACE-Step export script.

Reads the processed results (results.json) and the original audio files,
then produces two output directories ready for ACE-Step 1.5 fine-tuning:

  output/vae_training/   — 48 kHz stereo FLAC audio only
  output/dit_training/   — 48 kHz stereo FLAC + .lyrics.txt + .json annotations

Usage:
    python export_acestep.py
    python export_acestep.py --results results.json --music-dir music_for_preprocessing --output output
"""

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from src.export.cleaners import clean_caption, clean_lyrics_for_acestep
from src.export.language import detect_language

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 48000
TARGET_CHANNELS = 2
MIN_DURATION = 30
MAX_DURATION = 600
MIN_BPM = 20
MAX_BPM = 300


# ---------------------------------------------------------------------------
# Audio helpers (ffmpeg)
# ---------------------------------------------------------------------------

def _probe_audio(path: Path) -> dict | None:
    """Return sample_rate, channels, duration via ffprobe. None on failure."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", "-show_format",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        info = json.loads(out.stdout)
        # Find the audio stream
        for s in info.get("streams", []):
            if s.get("codec_type") == "audio":
                sr = int(s.get("sample_rate", 0))
                ch = int(s.get("channels", 0))
                dur = float(info.get("format", {}).get("duration", 0))
                return {"sample_rate": sr, "channels": ch, "duration": dur}
    except Exception:
        return None
    return None


def _resample_to_flac(src: Path, dst: Path, probe: dict) -> bool:
    """Resample/convert audio to 48 kHz stereo FLAC. Returns True on success."""
    needs_resample = probe["sample_rate"] != TARGET_SAMPLE_RATE
    needs_remix = probe["channels"] != TARGET_CHANNELS

    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if needs_resample:
        cmd += ["-ar", str(TARGET_SAMPLE_RATE)]
    if needs_remix:
        cmd += ["-ac", str(TARGET_CHANNELS)]
    # 24-bit FLAC
    cmd += ["-sample_fmt", "s32", "-c:a", "flac", str(dst)]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as exc:
        log.warning("ffmpeg failed for %s: %s", src.name, exc.stderr[:300] if exc.stderr else "")
        return False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class _Stats:
    def __init__(self):
        self.total = 0
        self.exported = 0
        self.skipped_duration = 0
        self.skipped_bpm = 0
        self.skipped_caption = 0
        self.skipped_corrupt = 0
        self.warn_missing_lyrics = 0


def _parse_bpm(raw: str) -> int | None:
    """Convert BPM string to int, rounding. None if invalid."""
    try:
        val = float(raw)
        return round(val)
    except (ValueError, TypeError):
        return None


def _sanitize_stem(name: str, idx: int) -> str:
    """Create a clean filename stem: track_00001 style."""
    return f"track_{idx:05d}"


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------

def export(results_path: Path, music_dir: Path, output_dir: Path, manifest_path: Path | None = None):
    # Load results
    if not results_path.exists():
        log.error("Results file not found: %s", results_path)
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        results: list[dict] = json.load(f)

    # Also load manifest.csv to get has_vocals info (results.json doesn't have it)
    has_vocals_map: dict[str, bool] = {}
    if manifest_path and manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fn = row.get("file_name", row.get("filename", ""))
                hv = row.get("has_vocals", "true").strip().lower() != "false"
                has_vocals_map[fn] = hv

    vae_dir = output_dir / "vae_training"
    dit_dir = output_dir / "dit_training"
    vae_dir.mkdir(parents=True, exist_ok=True)
    dit_dir.mkdir(parents=True, exist_ok=True)

    stats = _Stats()
    vae_manifest_rows: list[dict] = []
    dit_manifest_rows: list[dict] = []

    for idx, entry in enumerate(results, 1):
        stats.total += 1

        # Skip entries that had errors during processing
        if "error" in entry:
            log.warning("Skipping errored entry: %s", entry.get("file_name", f"entry {idx}"))
            stats.skipped_corrupt += 1
            continue

        file_name = entry.get("file_name", "")
        song_name = entry.get("song_name", "")
        artist = entry.get("artist", "")
        raw_bpm = entry.get("bpm", "")
        raw_key = entry.get("key", "")
        raw_ts = entry.get("time_signature", "")
        raw_lyrics = entry.get("lyrics", "")
        raw_caption = entry.get("caption", "")

        audio_path = music_dir / file_name
        if not audio_path.exists():
            log.warning("Audio file not found, skipping: %s", audio_path)
            stats.skipped_corrupt += 1
            continue

        # --- Probe audio ---
        probe = _probe_audio(audio_path)
        if probe is None:
            log.warning("Cannot probe audio (corrupted?), skipping: %s", file_name)
            stats.skipped_corrupt += 1
            continue

        duration = probe["duration"]

        # --- Validate duration ---
        if duration < MIN_DURATION or duration > MAX_DURATION:
            log.warning("Duration %.1fs out of range [%d-%d], skipping: %s",
                        duration, MIN_DURATION, MAX_DURATION, file_name)
            stats.skipped_duration += 1
            continue

        # --- Validate BPM ---
        bpm_int = _parse_bpm(raw_bpm)
        if bpm_int is None or bpm_int < MIN_BPM or bpm_int > MAX_BPM:
            log.warning("BPM '%s' invalid or out of range, skipping: %s", raw_bpm, file_name)
            stats.skipped_bpm += 1
            continue

        # --- Clean caption ---
        caption = clean_caption(raw_caption)
        if not caption:
            log.warning("Caption empty after cleanup, skipping: %s", file_name)
            stats.skipped_caption += 1
            continue

        # --- Determine has_vocals ---
        has_vocals = has_vocals_map.get(file_name, True)

        # --- Clean lyrics ---
        cleaned_lyrics = clean_lyrics_for_acestep(raw_lyrics)

        # Prepare lyrics for file
        if not has_vocals:
            lyrics_content = "[Instrumental]\n(instrumental)"
        elif not cleaned_lyrics:
            lyrics_content = ""
            if has_vocals:
                stats.warn_missing_lyrics += 1
                log.info("No lyrics for vocal track: %s", file_name)
        else:
            lyrics_content = cleaned_lyrics

        # --- Language detection ---
        language = detect_language(cleaned_lyrics, has_vocals)

        # --- ACE-Step field mapping ---
        keyscale = raw_key  # e.g. "C# minor", "F major"
        timesignature = raw_ts  # e.g. "4/4", "3/4"

        # --- Generate clean stem ---
        stem = _sanitize_stem(file_name, idx)

        # --- Resample and write audio to VAE dir ---
        vae_flac = vae_dir / f"{stem}.flac"
        if not _resample_to_flac(audio_path, vae_flac, probe):
            log.warning("Resampling failed, skipping: %s", file_name)
            stats.skipped_corrupt += 1
            continue

        # Get actual output duration from resampled file
        out_probe = _probe_audio(vae_flac)
        out_duration = out_probe["duration"] if out_probe else duration

        # --- Create hardlink/copy in DiT dir ---
        dit_flac = dit_dir / f"{stem}.flac"
        try:
            if dit_flac.exists():
                dit_flac.unlink()
            os.link(str(vae_flac), str(dit_flac))
        except OSError:
            # Hardlink not supported (cross-device etc.) — copy instead
            import shutil
            shutil.copy2(str(vae_flac), str(dit_flac))

        # --- Write lyrics file ---
        dit_lyrics = dit_dir / f"{stem}.lyrics.txt"
        dit_lyrics.write_text(lyrics_content, encoding="utf-8")

        # --- Write annotation JSON ---
        annotation = {
            "caption": caption,
            "bpm": bpm_int,
            "keyscale": keyscale,
            "timesignature": timesignature,
            "language": language,
        }
        dit_json = dit_dir / f"{stem}.json"
        dit_json.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")

        # --- Track manifest rows ---
        vae_manifest_rows.append({
            "filename": f"{stem}.flac",
            "duration_sec": round(out_duration, 1),
            "sample_rate": TARGET_SAMPLE_RATE,
            "channels": TARGET_CHANNELS,
        })
        dit_manifest_rows.append({
            "filename": f"{stem}.flac",
            "song_name": song_name,
            "artist": artist,
            "genre": "",
            "bpm": bpm_int,
            "keyscale": keyscale,
            "timesignature": timesignature,
            "language": language,
            "duration_sec": round(out_duration, 1),
            "has_vocals": has_vocals,
            "has_lyrics": bool(lyrics_content and lyrics_content != "[Instrumental]\n(instrumental)" and lyrics_content.strip()),
        })

        stats.exported += 1
        log.info("[%d/%d] Exported: %s → %s", idx, len(results), file_name, stem)

    # --- Write manifests ---
    _write_csv(vae_dir / "manifest.csv", vae_manifest_rows,
               ["filename", "duration_sec", "sample_rate", "channels"])
    _write_csv(dit_dir / "manifest.csv", dit_manifest_rows,
               ["filename", "song_name", "artist", "genre", "bpm", "keyscale",
                "timesignature", "language", "duration_sec", "has_vocals", "has_lyrics"])

    # --- Summary ---
    print(f"""
Export complete:
  Total tracks processed: {stats.total:,}
  Exported to vae_training: {stats.exported:,}
  Exported to dit_training: {stats.exported:,}
  Skipped (too short/long): {stats.skipped_duration:,}
  Skipped (invalid BPM): {stats.skipped_bpm:,}
  Skipped (empty caption): {stats.skipped_caption:,}
  Skipped (corrupted audio): {stats.skipped_corrupt:,}
  Warnings (missing lyrics for vocal tracks): {stats.warn_missing_lyrics:,}
""")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export processed data to ACE-Step training format")
    parser.add_argument("--results", default="results.json",
                        help="Path to results.json from process_songs.py (default: results.json)")
    parser.add_argument("--music-dir", default="music_for_preprocessing",
                        help="Directory containing source audio files (default: music_for_preprocessing)")
    parser.add_argument("--manifest", default="music_for_preprocessing/manifest.csv",
                        help="Path to manifest.csv for has_vocals info (default: music_for_preprocessing/manifest.csv)")
    parser.add_argument("--output", default="output",
                        help="Output base directory (default: output)")
    args = parser.parse_args()

    export(
        results_path=Path(args.results),
        music_dir=Path(args.music_dir),
        output_dir=Path(args.output),
        manifest_path=Path(args.manifest),
    )


if __name__ == "__main__":
    main()
