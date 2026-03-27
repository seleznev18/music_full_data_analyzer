#!/usr/bin/env python3
"""
High-throughput async pipeline for music metadata processing.

3-stage pipeline:
  Stage 1: Download audio files from S3 (AWS CLI subprocess)
  Stage 2: Fetch lyrics from Genius API (aiohttp)
  Stage 3: Generate captions via Gemini API / kie.ai proxy (aiohttp SSE)

Output: results.jsonl — one JSON object per line, crash-safe and resumable.

Usage:
    python3 pipeline.py
    python3 pipeline.py --manifest manifest.csv --output results.jsonl
    python3 pipeline.py --download-workers 20 --gemini-concurrency 8
"""

import argparse
import asyncio
import base64
import csv
import itertools
import json
import logging
import os
import random
import re
import signal
import sys
import time
import traceback
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════
# Environment
# ═══════════════════════════════════════════════════════════════════════════
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash")

# Support multiple Genius keys — comma-separated in GENIUS_API_TOKENS,
# falls back to single GENIUS_API_TOKEN.  Round-robined across requests.
_genius_tokens: list[str] = [
    t.strip()
    for t in os.getenv("GENIUS_API_TOKENS", os.getenv("GENIUS_API_TOKEN", "")).split(",")
    if t.strip()
]
_genius_token_cycle = itertools.cycle(_genius_tokens) if _genius_tokens else itertools.cycle([""])

# ═══════════════════════════════════════════════════════════════════════════
# Error logger — all errors with tracebacks go to pipeline_errors.log
# ═══════════════════════════════════════════════════════════════════════════
error_logger = logging.getLogger("pipeline_errors")
_err_handler = logging.FileHandler("pipeline_errors.log", mode="a", encoding="utf-8")
_err_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
error_logger.addHandler(_err_handler)
error_logger.setLevel(logging.ERROR)

# ── Lyrics debug logger — one line per song showing what happened ──
lyrics_logger = logging.getLogger("lyrics_debug")
_lyrics_handler = logging.FileHandler("lyrics_debug.log", mode="a", encoding="utf-8")
_lyrics_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
lyrics_logger.addHandler(_lyrics_handler)
lyrics_logger.setLevel(logging.DEBUG)

# ═══════════════════════════════════════════════════════════════════════════
# Caption prompt — exact copy from src/caption/gemini_service.py
# ═══════════════════════════════════════════════════════════════════════════
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
GENIUS_BASE_URL = "https://api.genius.com"


# ═══════════════════════════════════════════════════════════════════════════
# Caption cleaning — exact copy from src/export/cleaners.py
# ═══════════════════════════════════════════════════════════════════════════

_GARBAGE_PATTERNS = [
    "Is there anything else",
    "Let me know if",
    "I hope this helps",
    "Feel free to ask",
    "Would you like me to",
    "Here's the description",
    "Here is the description",
    "Here's a description",
    "Here is a description",
    "Sure, here",
    "Sure! Here",
    "Of course!",
    "Certainly!",
    "I'd be happy to",
    "Do you want me to",
    "I can also",
    "If you need",
    "Hope that helps",
    "Don't hesitate to",
    "Happy to help",
    "I've described",
    "I have described",
    "This description captures",
    "This should capture",
]

_LEADING_FLUFF = re.compile(
    r"^(Sure[:\!]?\s*|Here[:\!]?\s*|Description[:\s]*|Of course[:\!]?\s*|Certainly[:\!]?\s*)",
    re.IGNORECASE,
)

_MARKDOWN = re.compile(r"(\*{1,2}|#{1,6}\s?|`{1,3})")


def clean_caption(caption: str) -> str:
    """Remove Gemini conversational garbage from captions."""
    if not caption:
        return ""

    paragraphs = caption.split("\n\n")
    cleaned_paragraphs: list[str] = []
    for para in paragraphs:
        para_stripped = para.strip()
        if not para_stripped:
            continue
        is_garbage = any(
            pat.lower() in para_stripped.lower() for pat in _GARBAGE_PATTERNS
        )
        if not is_garbage:
            cleaned_paragraphs.append(para_stripped)

    result = "\n\n".join(cleaned_paragraphs)

    # Remove markdown formatting
    result = _MARKDOWN.sub("", result)

    # Strip wrapping quotes if the entire text is quoted
    if len(result) >= 2 and result[0] == result[-1] and result[0] in ('"', "'"):
        result = result[1:-1]

    # Remove leading conversational fluff
    result = _LEADING_FLUFF.sub("", result).strip()

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Lyrics cleaning — exact copy from src/lyrics/genius_provider.py
# ═══════════════════════════════════════════════════════════════════════════

def _clean_lyrics(lyrics: str) -> str:
    """Strip Genius page artifacts (contributors, translations, song description)."""
    # Find the first section marker like [Verse 1], [Intro], [Chorus], etc.
    marker = re.search(
        r'\[(Verse|Chorus|Intro|Pre-Chorus|Bridge|Outro|Hook|Interlude|Refrain|Part|Skit)',
        lyrics,
    )
    if marker:
        lyrics = lyrics[marker.start():]
    else:
        return ""

    # Remove trailing "You might also likeN Embed" type artifacts
    lyrics = re.sub(r'You might also like.*$', '', lyrics, flags=re.DOTALL)
    lyrics = re.sub(r'\d*\s*Embed\s*$', '', lyrics)

    return lyrics.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Genius lyrics — async port of src/lyrics/genius_provider.py
#
# Same URL construction, same headers, same response parsing,
# same HTML selectors, same text cleaning.  Only the HTTP client
# changed: httpx.Client → aiohttp.ClientSession.
# ═══════════════════════════════════════════════════════════════════════════

class _RateLimited(Exception):
    """Raised when Genius returns HTTP 429."""


async def _genius_search(
    session: aiohttp.ClientSession, title: str, artist: str,
) -> str | None:
    """Search Genius API for the song page URL.  Returns URL or None."""
    query = f"{title} {artist}"
    headers = {"Authorization": f"Bearer {next(_genius_token_cycle)}"}

    async with session.get(
        f"{GENIUS_BASE_URL}/search",
        params={"q": query},
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=15),
    ) as resp:
        if resp.status == 429:
            raise _RateLimited()
        resp.raise_for_status()
        data = await resp.json()

    hits = data.get("response", {}).get("hits", [])
    if not hits:
        return None

    artist_lower = artist.lower()
    for hit in hits:
        result = hit.get("result", {})
        primary_artist = result.get("primary_artist", {}).get("name", "").lower()
        if artist_lower in primary_artist or primary_artist in artist_lower:
            return result["url"]

    # Fallback to first result if no exact artist match
    return hits[0]["result"]["url"]


async def _genius_scrape(session: aiohttp.ClientSession, url: str) -> str:
    """Scrape lyrics from a Genius song page.  Returns cleaned lyrics."""
    async with session.get(
        url,
        timeout=aiohttp.ClientTimeout(total=15),
        allow_redirects=True,
    ) as resp:
        if resp.status == 429:
            raise _RateLimited()
        resp.raise_for_status()
        html = await resp.text()

    soup = BeautifulSoup(html, "html.parser")
    containers = soup.select('div[data-lyrics-container="true"]')

    if not containers:
        lyrics_logger.debug("NO_CONTAINER  url=%s", url)
        return ""

    parts: list[str] = []
    for container in containers:
        for br in container.find_all("br"):
            br.replace_with("\n")
        parts.append(container.get_text())

    lyrics = "\n".join(parts).strip()
    lyrics = re.sub(r"\n{3,}", "\n\n", lyrics)
    raw_len = len(lyrics)
    lyrics = _clean_lyrics(lyrics)
    if not lyrics:
        lyrics_logger.debug("CLEAN_EMPTY   url=%s  raw_len=%d", url, raw_len)
    return lyrics


async def fetch_lyrics(
    session: aiohttp.ClientSession,
    title: str,
    artist: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Fetch lyrics with Genius rate-limit retry.  Returns lyrics or empty string."""
    backoff_base = 2
    max_backoff = 20
    max_retries = 5

    for attempt in range(max_retries):
        try:
            async with semaphore:
                song_url = await _genius_search(session, title, artist)
                if song_url is None:
                    lyrics_logger.debug("NOT_FOUND     artist=%r  title=%r", artist, title)
                    return ""
                lyrics = await _genius_scrape(session, song_url)
                if lyrics:
                    lyrics_logger.debug("OK (%d chars)  artist=%r  title=%r  url=%s", len(lyrics), artist, title, song_url)
                else:
                    lyrics_logger.debug("EMPTY         artist=%r  title=%r  url=%s", artist, title, song_url)
                return lyrics
        except _RateLimited:
            wait = min(backoff_base * (2 ** attempt), max_backoff)
            await asyncio.sleep(wait)
            continue
        except Exception:
            lyrics_logger.debug("EXCEPTION     artist=%r  title=%r  err=%s", artist, title, traceback.format_exc().strip().splitlines()[-1])
            return ""

    lyrics_logger.debug("RATELIMITED   artist=%r  title=%r  (exhausted %d retries)", artist, title, max_retries)
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# Gemini caption — async port of src/caption/gemini_service.py
#
# Same endpoint URL, same request body structure (messages with
# text + image_url data-URI), same streaming SSE parsing, same
# ffmpeg quality setting (-q:a 2).
# ═══════════════════════════════════════════════════════════════════════════

async def _to_mp3_bytes(audio_path: str) -> bytes | None:
    """Convert audio to MP3 bytes.  MP3 files are read directly;
    everything else goes through ffmpeg with -q:a 2 (matching existing code)."""
    if audio_path.lower().endswith(".mp3"):
        try:
            with open(audio_path, "rb") as f:
                return f.read()
        except Exception as exc:
            error_logger.error("MP3 read failed: %s — %s", audio_path, exc)
            return None

    MAX_CAPTION_SECS = 300   # first 5 minutes
    MAX_MP3_BYTES    = 4_500_000  # ~4.5 MB

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-i", audio_path,
        "-t", str(MAX_CAPTION_SECS),
        "-map", "0:a:0",          # audio stream only, drop album art
        "-map_metadata", "-1",    # strip all metadata/ID3 tags
        "-ac", "1",               # mono — halves file size vs stereo
        "-b:a", "96k",            # CBR 96 kbps — predictable size
        "-f", "mp3", "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr_bytes = await proc.communicate()
    if proc.returncode != 0:
        error_logger.error(
            "ffmpeg failed (rc=%d): %s — %s",
            proc.returncode, audio_path,
            stderr_bytes.decode("utf-8", errors="replace")[:500],
        )
        return None
    if len(stdout) > MAX_MP3_BYTES:
        error_logger.error(
            "MP3 too large after conversion (%d bytes): %s",
            len(stdout), audio_path,
        )
        return None
    return stdout


async def generate_caption(
    session: aiohttp.ClientSession,
    audio_path: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Generate a caption via Gemini (kie.ai proxy) with retry logic.
    Returns the cleaned caption or empty string on failure."""

    # Convert to MP3 once (outside retry loop)
    mp3_bytes = await _to_mp3_bytes(audio_path)
    if not mp3_bytes:
        error_logger.error("Caption skip (mp3 conversion failed): %s", audio_path)
        return ""

    b64_data = base64.b64encode(mp3_bytes).decode("utf-8")
    del mp3_bytes  # free ~5 MB
    data_uri = f"data:audio/mpeg;base64,{b64_data}"

    url = f"{KIE_BASE_URL}/{GEMINI_MODEL}/v1/chat/completions"

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
        "Authorization": f"Bearer {GEMINI_API_KEY}",
    }

    gemini_timeout = aiohttp.ClientTimeout(connect=30, total=300)

    retries_429 = 0
    retries_server = 0

    for attempt in range(10):
        _retry_wait: float = 0.0  # sleep OUTSIDE semaphore to free the slot
        try:
            async with semaphore:
                async with session.post(
                    url, json=payload, headers=headers, timeout=gemini_timeout,
                ) as resp:
                    if resp.status == 429:
                        retries_429 += 1
                        if retries_429 > 10:
                            error_logger.error("Caption skip (429 exhausted after %d retries): %s", retries_429, audio_path)
                            return ""
                        _retry_wait = min(5 * (2 ** (retries_429 - 1)), 120) + random.uniform(0, 5)

                    elif resp.status >= 500:
                        retries_server += 1
                        if retries_server > 8:
                            error_logger.error("Caption skip (server %d exhausted after %d retries): %s", resp.status, retries_server, audio_path)
                            return ""
                        _retry_wait = min(3 * (2 ** (retries_server - 1)), 60) + random.uniform(0, 3)

                    elif resp.status >= 400:
                        body = await resp.text()
                        error_logger.error("Caption skip (HTTP %d): %s — %s", resp.status, audio_path, body[:500])
                        return ""

                    else:
                        # ── Read full response body, then parse ──
                        body = await resp.text()

                        # kie.ai wraps errors as HTTP 200 with {"code":NNN} body
                        soft_code = None
                        try:
                            _err = json.loads(body)
                            if isinstance(_err, dict) and "code" in _err and _err.get("code") != 200:
                                soft_code = _err["code"]
                        except (json.JSONDecodeError, TypeError):
                            pass

                        if soft_code == 429:
                            retries_429 += 1
                            if retries_429 > 10:
                                error_logger.error("Caption skip (soft-429 exhausted after %d retries): %s", retries_429, audio_path)
                                return ""
                            _retry_wait = min(5 * (2 ** (retries_429 - 1)), 120) + random.uniform(0, 5)

                        elif soft_code in (500, 502, 503):
                            retries_server += 1
                            if retries_server > 8:
                                error_logger.error("Caption skip (soft-%d exhausted after %d retries): %s", soft_code, retries_server, audio_path)
                                return ""
                            _retry_wait = min(3 * (2 ** (retries_server - 1)), 60) + random.uniform(0, 3)

                        elif soft_code is not None:
                            # Any other error code (400, 401, etc.) — not retryable
                            error_logger.error("Caption skip (soft-%d): %s — %s", soft_code, audio_path, body[:500])
                            return ""

                        else:
                            content_parts: list[str] = []

                            # Try SSE parsing first (lines starting with "data:")
                            for line in body.splitlines():
                                line = line.strip()
                                if not line.startswith("data:"):
                                    continue
                                data_str = line[len("data:"):].strip()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                except json.JSONDecodeError:
                                    continue
                                choices = chunk.get("choices", [])
                                if not choices:
                                    continue
                                delta = choices[0].get("delta", {})
                                text = delta.get("content", "")
                                if text:
                                    content_parts.append(text)

                            # Fallback: try parsing body as regular JSON response
                            if not content_parts:
                                try:
                                    full_resp = json.loads(body)
                                    choices = full_resp.get("choices", [])
                                    if choices:
                                        msg = choices[0].get("message", {})
                                        text = msg.get("content", "")
                                        if text:
                                            content_parts.append(text)
                                except (json.JSONDecodeError, KeyError, IndexError):
                                    pass

                            if not content_parts:
                                error_logger.error("Caption skip (empty response, status %d): %s — body: %.500s", resp.status, audio_path, body)
                                return ""

                            raw_caption = "".join(content_parts)
                            cleaned = clean_caption(raw_caption)
                            if not cleaned:
                                error_logger.error("Caption skip (clean_caption stripped everything): %s — raw was: %.200s", audio_path, raw_caption)
                            return cleaned

        except asyncio.TimeoutError:
            retries_server += 1
            if retries_server > 5:
                error_logger.error("Caption skip (timeout exhausted after %d retries): %s", retries_server, audio_path)
                return ""
            _retry_wait = 3.0 + random.uniform(0, 2)
        except aiohttp.ClientResponseError as exc:
            error_logger.error("Caption skip (HTTP error %s): %s", exc, audio_path)
            return ""
        except Exception:
            error_logger.error("Caption skip (unexpected error): %s\n%s", audio_path, traceback.format_exc())
            return ""

        # Sleep AFTER releasing the semaphore so the slot is free for other workers
        if _retry_wait > 0:
            await asyncio.sleep(_retry_wait)

    error_logger.error("Caption skip (all 10 attempts exhausted): %s", audio_path)
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# S3 Download (AWS CLI subprocess)
# ═══════════════════════════════════════════════════════════════════════════

async def download_from_s3(
    filename: str,
    temp_dir: str,
    bucket: str,
    prefix: str,
    endpoint: str,
) -> str | None:
    """Download a file from S3 using the AWS CLI.
    Returns local file path on success, or the error string on failure."""
    local_path = os.path.join(temp_dir, filename)
    s3_uri = f"s3://{bucket}/{prefix}/{filename}"

    last_stderr = ""
    for attempt in range(3):
        proc = await asyncio.create_subprocess_exec(
            "aws", "s3", "--endpoint-url", endpoint, "cp", s3_uri, local_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_bytes = await proc.communicate()
        if proc.returncode == 0:
            return local_path
        last_stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
        if attempt < 2:
            await asyncio.sleep(2)

    return last_stderr  # return error message instead of None on failure


# ═══════════════════════════════════════════════════════════════════════════
# Resume — load already-processed file_ids from existing JSONL
# ═══════════════════════════════════════════════════════════════════════════

def load_processed_ids(output_path: str) -> set[str]:
    """Read existing results.jsonl and return a set of file_ids already done."""
    processed: set[str] = set()
    if not os.path.exists(output_path):
        return processed
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    fid = obj.get("file_id", "")
                    if fid:
                        processed.add(fid)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return processed


# ═══════════════════════════════════════════════════════════════════════════
# Manifest reading
# ═══════════════════════════════════════════════════════════════════════════

def read_manifest(manifest_path: str) -> list[dict]:
    """Read manifest CSV.  Strips whitespace from all values.
    Falls back to latin-1 if UTF-8 decoding fails."""
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(manifest_path, newline="", encoding=encoding) as f:
                rows = []
                for row in csv.DictReader(f):
                    stripped = {k.strip(): v.strip() for k, v in row.items()}
                    rows.append(stripped)
                return rows
        except UnicodeDecodeError:
            continue
    return []


def validate_row(row: dict) -> bool:
    """Check that a manifest row has all required fields."""
    return bool(
        row.get("file_id")
        and row.get("filename")
        and row.get("song_name")
        and row.get("artist")
    )


# ═══════════════════════════════════════════════════════════════════════════
# Counters (safe in single-threaded async — no locks needed)
# ═══════════════════════════════════════════════════════════════════════════

class Counters:
    def __init__(self):
        self.downloaded = 0
        self.lyrics_done = 0
        self.captioned = 0
        self.failed = 0
        self.skipped = 0


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: S3 Download workers
# ═══════════════════════════════════════════════════════════════════════════

async def stage1_worker(
    manifest_queue: asyncio.Queue,
    queue_a: asyncio.Queue,
    args: argparse.Namespace,
    counters: Counters,
):
    """Download worker: pulls manifest rows, downloads from S3,
    pushes results into queue_a."""
    while True:
        item = await manifest_queue.get()
        if item is None:
            return  # sentinel — this worker is done

        row = item
        file_id = row["file_id"]
        filename = row["filename"]

        try:
            result = await download_from_s3(
                filename, args.temp_dir,
                args.s3_bucket, args.s3_prefix, args.s3_endpoint,
            )
            local_path = os.path.join(args.temp_dir, filename)
            if not os.path.exists(local_path):
                error_logger.error(
                    "S3 download failed after 3 retries: file_id=%s filename=%s — %s",
                    file_id, filename, result,
                )
                counters.failed += 1
                continue

            counters.downloaded += 1

            await queue_a.put({
                "file_id": file_id,
                "file_name": filename,
                "local_path": local_path,
                "song_name": row["song_name"],
                "artist": row["artist"],
                "bpm": row.get("bpm", ""),
                "key": row.get("key", ""),
                "time_signature": row.get("time_signature", ""),
                "has_vocals": row.get("has_vocals", "true").lower() != "false",
            })
        except Exception:
            error_logger.error(
                "Stage 1 error: file_id=%s\n%s", file_id, traceback.format_exc()
            )
            counters.failed += 1


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Genius Lyrics workers
# ═══════════════════════════════════════════════════════════════════════════

async def stage2_worker(
    queue_a: asyncio.Queue,
    queue_b: asyncio.Queue,
    session: aiohttp.ClientSession,
    genius_semaphore: asyncio.Semaphore,
    counters: Counters,
):
    """Lyrics worker: pulls items from queue_a, fetches lyrics via Genius,
    pushes results into queue_b."""
    while True:
        item = await queue_a.get()
        if item is None:
            return  # sentinel

        file_id = item["file_id"]

        try:
            lyrics = ""
            if item["has_vocals"]:
                lyrics = await fetch_lyrics(
                    session, item["song_name"], item["artist"], genius_semaphore,
                )
            counters.lyrics_done += 1

            await queue_b.put({
                "file_id": item["file_id"],
                "file_name": item["file_name"],
                "local_path": item["local_path"],
                "song_name": item["song_name"],
                "artist": item["artist"],
                "bpm": item["bpm"],
                "key": item["key"],
                "time_signature": item["time_signature"],
                "has_vocals": item["has_vocals"],
                "lyrics": lyrics,
            })
        except Exception:
            error_logger.error(
                "Stage 2 error: file_id=%s\n%s", file_id, traceback.format_exc()
            )
            counters.lyrics_done += 1
            # Pass through with empty lyrics so Stage 3 can still caption
            await queue_b.put({
                "file_id": item["file_id"],
                "file_name": item["file_name"],
                "local_path": item["local_path"],
                "song_name": item["song_name"],
                "artist": item["artist"],
                "bpm": item["bpm"],
                "key": item["key"],
                "time_signature": item["time_signature"],
                "has_vocals": item["has_vocals"],
                "lyrics": "",
            })


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3: Gemini Caption workers
# ═══════════════════════════════════════════════════════════════════════════

async def stage3_worker(
    queue_b: asyncio.Queue,
    queue_c: asyncio.Queue,
    session: aiohttp.ClientSession,
    gemini_semaphore: asyncio.Semaphore,
    counters: Counters,
):
    """Caption worker: pulls items from queue_b, generates captions via
    Gemini, pushes final results into queue_c.  Deletes temp file after."""
    while True:
        item = await queue_b.get()
        if item is None:
            return  # sentinel

        file_id = item["file_id"]
        local_path = item.pop("local_path")

        try:
            caption = await generate_caption(session, local_path, gemini_semaphore)

            if caption:
                counters.captioned += 1
            else:
                counters.failed += 1

            item["caption"] = caption
            await queue_c.put(item)

        except Exception:
            error_logger.error(
                "Stage 3 error: file_id=%s\n%s", file_id, traceback.format_exc()
            )
            counters.failed += 1
            item["caption"] = ""
            await queue_c.put(item)

        finally:
            # Always clean up temp file to free disk space
            try:
                if os.path.exists(local_path):
                    os.unlink(local_path)
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# Writer coroutine — appends to JSONL, flushes after every write
# ═══════════════════════════════════════════════════════════════════════════

async def writer_worker(
    queue_c: asyncio.Queue,
    output_path: str,
    pbar: tqdm,
):
    """Reads final results from queue_c and appends them to the JSONL file."""
    with open(output_path, "a", encoding="utf-8") as f:
        while True:
            item = await queue_c.get()
            if item is None:
                return  # sentinel — done

            line = json.dumps(item, ensure_ascii=False)
            f.write(line + "\n")
            f.flush()
            pbar.update(1)


# ═══════════════════════════════════════════════════════════════════════════
# Status monitor — periodic stats to stderr every 60 seconds
# ═══════════════════════════════════════════════════════════════════════════

async def status_monitor(
    counters: Counters,
    queue_a: asyncio.Queue,
    queue_b: asyncio.Queue,
    temp_dir: str,
    stop_event: asyncio.Event,
):
    """Log pipeline status every 60 seconds until stop_event is set."""
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=60)
            return  # stop_event was set
        except asyncio.TimeoutError:
            pass  # 60 seconds elapsed — print status

        try:
            temp_file_count = len(os.listdir(temp_dir))
        except OSError:
            temp_file_count = -1

        print(
            f"\n[STATUS] Downloaded: {counters.downloaded} | "
            f"Lyrics: {counters.lyrics_done} | "
            f"Captioned: {counters.captioned} | "
            f"Failed: {counters.failed} | "
            f"Queue A: ~{queue_a.qsize()} | "
            f"Queue B: ~{queue_b.qsize()} | "
            f"Temp files on disk: {temp_file_count}",
            file=sys.stderr,
            flush=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline orchestrator
# ═══════════════════════════════════════════════════════════════════════════

async def run_pipeline(args: argparse.Namespace):
    # Ensure temp directory exists
    os.makedirs(args.temp_dir, exist_ok=True)

    # ── Resume: load already-processed IDs ──
    processed_ids = load_processed_ids(args.output)

    # ── Read manifest and filter ──
    all_rows = read_manifest(args.manifest)
    total_in_manifest = len(all_rows)

    rows_to_process: list[dict] = []
    skipped_invalid = 0
    for row in all_rows:
        file_id = row.get("file_id", "")
        if file_id in processed_ids:
            continue
        if not validate_row(row):
            error_logger.error("Invalid manifest row (skipped): %s", row)
            skipped_invalid += 1
            continue
        rows_to_process.append(row)

    already_processed = len(processed_ids)
    remaining = len(rows_to_process)

    # ── Startup banner ──
    print("=" * 64)
    print("  Music Processing Pipeline")
    print("=" * 64)
    print(f"  Total files in manifest:   {total_in_manifest:,}")
    print(f"  Already processed:         {already_processed:,}")
    print(f"  Skipped (invalid rows):    {skipped_invalid:,}")
    print(f"  Remaining to process:      {remaining:,}")
    print(f"  ─────────────────────────────────")
    print(f"  Download workers:          {args.download_workers}")
    print(f"  Genius workers:            {args.genius_workers}")
    print(f"  Genius concurrency:        {args.genius_concurrency}")
    print(f"  Gemini workers:            {args.gemini_workers}")
    print(f"  Gemini concurrency:        {args.gemini_concurrency}")
    print(f"  Queue A maxsize:           {args.queue_a_size}")
    print(f"  Queue B maxsize:           {args.queue_b_size}")
    print(f"  Output:                    {args.output}")
    print(f"  Temp dir:                  {args.temp_dir}")
    print("=" * 64, flush=True)

    if remaining == 0:
        print("\nNothing to process. All files already in output.")
        return

    # ── Queues ──
    manifest_queue: asyncio.Queue = asyncio.Queue()            # unbounded — holds metadata only
    queue_a: asyncio.Queue = asyncio.Queue(maxsize=args.queue_a_size)  # download → lyrics
    queue_b: asyncio.Queue = asyncio.Queue(maxsize=args.queue_b_size)  # lyrics → caption
    queue_c: asyncio.Queue = asyncio.Queue()                           # caption → writer

    # ── Shutdown handling ──
    stop_event = asyncio.Event()
    _first_signal = [True]

    def _handle_signal():
        if not _first_signal[0]:
            print("\nForce shutdown.", file=sys.stderr, flush=True)
            os._exit(1)
        _first_signal[0] = False
        stop_event.set()
        print(
            "\nGraceful shutdown requested — finishing in-progress work. "
            "Press Ctrl+C again to force quit.",
            file=sys.stderr, flush=True,
        )

    loop = asyncio.get_event_loop()
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _handle_signal)
    except NotImplementedError:
        pass  # Windows — KeyboardInterrupt will handle it

    # ── Counters and progress bar ──
    counters = Counters()
    pbar = tqdm(total=remaining, desc="Processing", unit="file")

    N = args.download_workers
    M = args.genius_workers
    K = args.gemini_workers

    # ── aiohttp session ──
    session_timeout = aiohttp.ClientTimeout(connect=30, total=180)
    async with aiohttp.ClientSession(timeout=session_timeout) as session:

        # Start status monitor
        monitor_task = asyncio.create_task(
            status_monitor(counters, queue_a, queue_b, args.temp_dir, stop_event)
        )

        # Start writer
        writer_task = asyncio.create_task(
            writer_worker(queue_c, args.output, pbar)
        )

        # Start Stage 3 (caption) workers
        gemini_semaphore = asyncio.Semaphore(args.gemini_concurrency)
        caption_tasks = [
            asyncio.create_task(stage3_worker(
                queue_b, queue_c, session, gemini_semaphore, counters,
            ))
            for _ in range(K)
        ]

        # Start Stage 2 (lyrics) workers
        genius_semaphore = asyncio.Semaphore(args.genius_concurrency)
        lyrics_tasks = [
            asyncio.create_task(stage2_worker(
                queue_a, queue_b, session, genius_semaphore, counters,
            ))
            for _ in range(M)
        ]

        # Start Stage 1 (download) workers
        download_tasks = [
            asyncio.create_task(stage1_worker(
                manifest_queue, queue_a, args, counters,
            ))
            for _ in range(N)
        ]

        # ── Feed manifest into manifest_queue ──
        for row in rows_to_process:
            if stop_event.is_set():
                break
            manifest_queue.put_nowait(row)

        # Signal download workers to stop (one sentinel per worker)
        for _ in range(N):
            manifest_queue.put_nowait(None)

        # ── Cascading shutdown: wait for each stage, then signal the next ──

        # Wait for all Stage 1 workers to finish
        await asyncio.gather(*download_tasks)

        # Signal Stage 2 workers to stop
        for _ in range(M):
            await queue_a.put(None)

        # Wait for all Stage 2 workers to finish
        await asyncio.gather(*lyrics_tasks)

        # Signal Stage 3 workers to stop
        for _ in range(K):
            await queue_b.put(None)

        # Wait for all Stage 3 workers to finish
        await asyncio.gather(*caption_tasks)

        # Signal writer to stop
        await queue_c.put(None)

        # Wait for writer to finish
        await writer_task

        # Stop the status monitor
        stop_event.set()
        await monitor_task

    pbar.close()

    # ── Final statistics ──
    print("\n" + "=" * 64)
    print("  Pipeline complete")
    print("=" * 64)
    print(f"  Downloaded:   {counters.downloaded:,}")
    print(f"  Lyrics:       {counters.lyrics_done:,}")
    print(f"  Captioned:    {counters.captioned:,}")
    print(f"  Failed:       {counters.failed:,}")
    print(f"  Skipped:      {counters.skipped:,}")
    print("=" * 64)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="High-throughput async pipeline for music processing",
    )
    parser.add_argument(
        "--manifest", default="manifest.csv",
        help="Path to the manifest CSV file (default: manifest.csv)",
    )
    parser.add_argument(
        "--s3-bucket", default="n2gaz71k8d",
        help="S3 bucket name (default: n2gaz71k8d)",
    )
    parser.add_argument(
        "--s3-prefix", default="flac-dataset",
        help="S3 key prefix / folder (default: flac-dataset)",
    )
    parser.add_argument(
        "--s3-endpoint", default="https://s3api-eu-cz-1.runpod.io",
        help="S3 endpoint URL (default: https://s3api-eu-cz-1.runpod.io)",
    )
    parser.add_argument(
        "--output", default="results.jsonl",
        help="Output JSONL file path (default: results.jsonl)",
    )
    parser.add_argument(
        "--temp-dir", default="/tmp/music_processing",
        help="Temp directory for downloaded files (default: /tmp/music_processing)",
    )
    parser.add_argument(
        "--download-workers", type=int, default=10,
        help="Number of concurrent S3 download coroutines (default: 10)",
    )
    parser.add_argument(
        "--genius-workers", type=int, default=15,
        help="Number of Stage 2 lyrics coroutines (default: 15)",
    )
    parser.add_argument(
        "--genius-concurrency", type=int, default=5,
        help="Max concurrent HTTP requests to Genius API (default: 5)",
    )
    parser.add_argument(
        "--gemini-workers", type=int, default=10,
        help="Number of Stage 3 caption coroutines (default: 10)",
    )
    parser.add_argument(
        "--gemini-concurrency", type=int, default=5,
        help="Max concurrent HTTP requests to Gemini API (default: 5)",
    )
    parser.add_argument(
        "--queue-a-size", type=int, default=150,
        help="Max size of download→lyrics queue (default: 150)",
    )
    parser.add_argument(
        "--queue-b-size", type=int, default=300,
        help="Max size of lyrics→caption queue (default: 300)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(run_pipeline(args))
    except KeyboardInterrupt:
        print(
            f"\nPipeline interrupted. Completed results are saved in {args.output}."
            f"\nRestart with the same command to resume from where you left off.",
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
