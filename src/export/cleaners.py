"""
Text cleaning utilities for ACE-Step export.

- clean_caption: removes Gemini conversational garbage from captions.
- clean_lyrics_for_acestep: cleans Genius.com lyrics for ACE-Step format.
"""

import re


# ---------------------------------------------------------------------------
# Caption cleanup
# ---------------------------------------------------------------------------

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

    # Split by double newlines — garbage is usually in the last paragraph
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


# ---------------------------------------------------------------------------
# Lyrics cleanup for ACE-Step
# ---------------------------------------------------------------------------

# Section tags that ACE-Step expects
_GOOD_SECTION_TAG = re.compile(
    r"^\["
    r"(Verse|Chorus|Pre-Chorus|Bridge|Outro|Intro|Hook|Interlude|Refrain|Rap|Spoken|Part|Skit)"
    r"[^\]]*\]",
    re.IGNORECASE,
)

# Tags to remove (Genius metadata)
_META_TAG = re.compile(
    r"^\[(Produced by|Written by|Featuring|feat\.|ft\.)[^\]]*\]",
    re.IGNORECASE,
)

_CONTRIBUTOR_LINE = re.compile(r"^\d+\s*Contributors?", re.IGNORECASE)
_EMBED_LINE = re.compile(r"^\d*\s*Embed\s*$", re.IGNORECASE)
_YOU_MIGHT = re.compile(r"^You might also like", re.IGNORECASE)
_TRANSLATIONS = re.compile(r"^Translations?\s*$", re.IGNORECASE)


def clean_lyrics_for_acestep(lyrics: str) -> str:
    """Clean Genius.com lyrics formatting for ACE-Step training."""
    if not lyrics:
        return ""

    lines = lyrics.splitlines()
    cleaned: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Skip known Genius artifacts
        if _CONTRIBUTOR_LINE.match(stripped):
            continue
        if _EMBED_LINE.match(stripped):
            continue
        if _YOU_MIGHT.match(stripped):
            continue
        if _TRANSLATIONS.match(stripped):
            continue
        if _META_TAG.match(stripped):
            continue

        cleaned.append(line)

    result = "\n".join(cleaned).strip()

    # Remove empty section headers (header followed immediately by another header or end)
    result = re.sub(
        r"\[[^\]]+\]\s*\n\s*(?=\[[^\]]+\])",
        "",
        result,
    )

    # Collapse 3+ consecutive blank lines to 2
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()
