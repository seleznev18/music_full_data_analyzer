"""
Audio feature extractors.

Copied from music_analyzer with addition of TimeSignatureAnalyzer.
"""

from abc import ABC, abstractmethod

import numpy as np


class IAudioAnalyzer(ABC):
    """Abstract base for all audio feature extractors."""

    @abstractmethod
    def analyze(self, audio: np.ndarray) -> dict[str, object]:
        """
        Extract one or more features from a mono float32 audio array.

        Parameters
        ----------
        audio:
            Mono audio signal as a 1-D numpy float32 array, normalised to [-1, 1].

        Returns
        -------
        dict whose keys map to result fields.
        """
        ...


class KeyAnalyzer(IAudioAnalyzer):
    """
    Extracts the musical key and scale using Essentia's KeyExtractor.

    Returns {"key": "<note> <scale>"}  e.g. {"key": "A minor"}.
    """

    def analyze(self, audio: np.ndarray) -> dict[str, object]:
        import essentia.standard as es

        key_extractor = es.KeyExtractor()
        key, scale, _strength = key_extractor(audio)
        return {"key": f"{key} {scale}"}


class BpmAnalyzer(IAudioAnalyzer):
    """
    Extracts the tempo in BPM using Essentia's PercivalBpmEstimator.

    Returns {"bpm": <float>} rounded to two decimal places.
    """

    def analyze(self, audio: np.ndarray) -> dict[str, object]:
        import essentia.standard as es

        bpm = es.PercivalBpmEstimator()(audio)
        return {"bpm": round(float(bpm), 2)}


class TimeSignatureAnalyzer(IAudioAnalyzer):
    """
    Estimates the time signature by analysing beat-level accent patterns.

    Uses RhythmExtractor2013 for beat positions and BeatsLoudness
    to detect periodic accent patterns (3/4 vs 4/4).

    Returns {"time_signature": "4/4"} or similar.
    """

    def analyze(self, audio: np.ndarray) -> dict[str, object]:
        import essentia.standard as es

        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)

        if len(beats) < 8:
            return {"time_signature": "4/4"}

        loudness, _ = es.BeatsLoudness(beats=beats.tolist())(audio)

        if len(loudness) < 8:
            return {"time_signature": "4/4"}

        # Test common meters: 3 (waltz) and 4 (standard)
        scores: dict[int, float] = {}
        for meter in [3, 4]:
            accent_hits = 0
            groups = 0
            for i in range(0, len(loudness) - meter + 1, meter):
                group = loudness[i : i + meter]
                if len(group) == meter:
                    # First beat of each group should be loudest (downbeat)
                    if group[0] >= max(group):
                        accent_hits += 1
                    groups += 1
            scores[meter] = accent_hits / groups if groups > 0 else 0.0

        best_meter = max(scores, key=scores.get)
        meter_map = {3: "3/4", 4: "4/4"}
        return {"time_signature": meter_map.get(best_meter, "4/4")}
