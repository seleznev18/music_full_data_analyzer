"""
FastAPI entry point for music_full_data_analyzer.

Single endpoint that accepts an audio file upload with song metadata,
runs the full analysis pipeline (audio features, lyrics, caption),
and returns a JSON result.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from config import settings
from src.audio_analysis.analyzers import (
    BpmAnalyzer,
    KeyAnalyzer,
    TimeSignatureAnalyzer,
)
from src.audio_analysis.constants import ALLOWED_AUDIO_EXTENSIONS
from src.audio_analysis.service import AudioAnalysisService, AudioLoader
from src.caption.gemini_service import GeminiCaptionService
from src.lyrics.genius_provider import GeniusLyricsProvider

app = FastAPI(title="Music Full Data Analyzer")


# --- Response schema ---


class AnalysisResult(BaseModel):
    file_name: str
    song_name: str
    artist: str
    bpm: str
    key: str
    time_signature: str
    lyrics: str
    caption: str


# --- Service singletons ---


def get_audio_service() -> AudioAnalysisService:
    loader = AudioLoader()
    analyzers = [KeyAnalyzer(), BpmAnalyzer(), TimeSignatureAnalyzer()]
    return AudioAnalysisService(loader, analyzers)


def get_lyrics_provider() -> GeniusLyricsProvider | None:
    if not settings.genius_api_token:
        return None
    return GeniusLyricsProvider(api_token=settings.genius_api_token)


def get_caption_service() -> GeminiCaptionService | None:
    if not settings.gemini_api_key:
        return None
    return GeminiCaptionService(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_model,
    )


# --- Endpoint ---


@app.post("/api/v1/analyze", response_model=AnalysisResult)
def analyze_song(
    file: UploadFile = File(...),
    song_name: str = Form(...),
    artist: str = Form(...),
):
    """
    Analyze a single audio file: extract BPM/key/time signature,
    fetch lyrics from Genius, generate a caption via Gemini.
    """
    # Validate extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
        )

    # Save upload to a temp file (Essentia and caption service need a file path)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        audio_path = Path(tmp_path)

        # 1) Audio analysis
        audio_service = get_audio_service()
        features = audio_service.analyze_file(audio_path)

        if "error" in features:
            raise HTTPException(
                status_code=422,
                detail=f"Audio analysis failed: {features['error']}",
            )

        # 2) Lyrics
        lyrics = ""
        lyrics_provider = get_lyrics_provider()
        if lyrics_provider:
            try:
                result = lyrics_provider.fetch_lyrics(song_name, artist)
                lyrics = result.lyrics
            except Exception as exc:
                lyrics = f"[lyrics unavailable: {exc}]"

        # 3) Caption
        caption = ""
        caption_service = get_caption_service()
        if caption_service:
            try:
                caption = caption_service.generate_caption(audio_path)
            except Exception as exc:
                caption = f"[caption unavailable: {exc}]"

        return AnalysisResult(
            file_name=file.filename or "",
            song_name=song_name,
            artist=artist,
            bpm=str(features.get("bpm", "")),
            key=str(features.get("key", "")),
            time_signature=str(features.get("time_signature", "")),
            lyrics=lyrics,
            caption=caption,
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
