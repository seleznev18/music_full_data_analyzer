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
    Estimates the time signature via onset detection + autocorrelation.

    Computes a frame-level onset strength curve (complex spectral difference),
    then autocorrelates it. Peaks at specific lag ratios reveal whether the
    accent pattern repeats every 3 or 4 beats, without relying on beat tracking.

    Returns {"time_signature": "4/4"} or similar.
    """

    FRAME_SIZE = 2048
    HOP_SIZE = 512

    def analyze(self, audio: np.ndarray) -> dict[str, object]:
        import essentia.standard as es
        from src.audio_analysis.constants import SAMPLE_RATE

        # 1) Get BPM to know the expected beat period in frames
        bpm = float(es.PercivalBpmEstimator()(audio))
        if bpm <= 0:
            return {"time_signature": "4/4"}

        beat_period_sec = 60.0 / bpm
        beat_period_frames = beat_period_sec / (self.HOP_SIZE / SAMPLE_RATE)

        # 2) Compute frame-level onset detection function
        w = es.Windowing(type="hann")
        fft = es.FFT(size=self.FRAME_SIZE)
        c2p = es.CartesianToPolar()
        onset = es.OnsetDetection(method="complex")

        onset_curve = []
        for frame in es.FrameGenerator(audio, frameSize=self.FRAME_SIZE, hopSize=self.HOP_SIZE):
            fft_result = fft(w(frame))
            mag, phase = c2p(fft_result)
            onset_curve.append(onset(mag, phase))

        if len(onset_curve) < int(beat_period_frames * 8):
            return {"time_signature": "4/4"}

        onset_signal = np.array(onset_curve, dtype=np.float32)

        # 3) Autocorrelation of the onset strength curve
        autocorr = es.AutoCorrelation()(onset_signal)

        # 4) Compare autocorrelation strength at meter-level lags
        #    For 3/4 time: strong peak at lag = 3 * beat_period
        #    For 4/4 time: strong peak at lag = 4 * beat_period
        scores: dict[int, float] = {}
        for meter in [3, 4]:
            lag = int(round(beat_period_frames * meter))
            if lag < len(autocorr):
                scores[meter] = float(autocorr[lag])
            else:
                scores[meter] = 0.0

        best_meter = max(scores, key=scores.get)
        meter_map = {3: "3/4", 4: "4/4"}
        return {"time_signature": meter_map.get(best_meter, "4/4")}
