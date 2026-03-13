class AudioLoadError(Exception):
    """Raised when an audio file cannot be decoded or loaded."""

    def __init__(self, filename: str, reason: str) -> None:
        self.filename = filename
        self.reason = reason
        super().__init__(f"Cannot load '{filename}': {reason}")


class AudioAnalysisError(Exception):
    """Raised when a feature extractor fails during analysis."""

    def __init__(self, filename: str, feature: str, reason: str) -> None:
        self.filename = filename
        self.feature = feature
        self.reason = reason
        super().__init__(f"Analysis of '{filename}' failed for '{feature}': {reason}")


class UnsupportedAudioFormatError(Exception):
    """Raised when a file extension is not in the allowed set."""

    def __init__(self, filename: str, extension: str) -> None:
        self.filename = filename
        self.extension = extension
        super().__init__(
            f"File '{filename}' has unsupported format '{extension}'."
        )
