"""
Language detection from lyrics text using langdetect.
"""

from langdetect import detect, LangDetectException


def detect_language(lyrics: str, has_vocals: bool) -> str:
    """Return ISO 639-1 language code detected from lyrics.

    Falls back to "en" for instrumentals, empty lyrics, or detection failure.
    """
    if not has_vocals:
        return "en"

    if not lyrics or not lyrics.strip():
        return "en"

    # Strip section tags before detection — they are English regardless of song language
    import re
    text = re.sub(r"\[[^\]]*\]", "", lyrics).strip()
    if not text:
        return "en"

    try:
        return detect(text)
    except LangDetectException:
        return "en"
