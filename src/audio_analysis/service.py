"""
Audio loading and analysis service.

Copied from music_analyzer, adapted for CLI usage (file-path based loading).
"""

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.audio_analysis.analyzers import IAudioAnalyzer
from src.audio_analysis.constants import SAMPLE_RATE
from src.audio_analysis.exceptions import AudioLoadError


class AudioLoader:
    """Decodes an audio file into a mono float32 numpy array via Essentia MonoLoader."""

    def load_from_path(self, file_path: Path) -> np.ndarray:
        """Load audio directly from a file path."""
        if not file_path.exists():
            raise AudioLoadError(file_path.name, "File not found")
        return self._decode(str(file_path), file_path.name)

    def load_from_bytes(self, audio_bytes: bytes, filename: str) -> np.ndarray:
        """Load audio from raw bytes (writes to temp file for Essentia)."""
        suffix = os.path.splitext(filename)[-1] or ".tmp"
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            return self._decode(tmp_path, filename)
        except AudioLoadError:
            raise
        except Exception as exc:
            raise AudioLoadError(filename, str(exc)) from exc
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def _decode(path: str, filename: str) -> np.ndarray:
        import essentia.standard as es

        try:
            loader = es.MonoLoader(filename=path, sampleRate=SAMPLE_RATE)
            return loader()
        except Exception as exc:
            raise AudioLoadError(filename, str(exc)) from exc


class AudioAnalysisService:
    """Orchestrates audio loading and feature extraction."""

    def __init__(
        self,
        loader: AudioLoader,
        analyzers: Sequence[IAudioAnalyzer],
    ) -> None:
        self._loader = loader
        self._analyzers = analyzers

    def analyze_file(self, file_path: Path) -> dict[str, object]:
        """
        Analyze a single audio file from disk.

        Returns a dict with keys: key, bpm, time_signature (and error if failed).
        """
        try:
            audio = self._loader.load_from_path(file_path)
            features: dict[str, object] = {}
            for analyzer in self._analyzers:
                features.update(analyzer.analyze(audio))
            return features
        except Exception as exc:
            return {"error": str(exc)}
