"""
Genius lyrics scraping.

Copied from music_lyrics_scraper/app/services/lyrics.py,
adapted from async to sync (httpx.Client) for CLI usage.
"""

import re
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup


@dataclass
class LyricsResult:
    title: str
    artist: str
    lyrics: str


class GeniusLyricsProvider:
    """Fetches lyrics via the Genius API + page scraping (synchronous)."""

    BASE_URL = "https://api.genius.com"

    def __init__(self, api_token: str):
        self._api_token = api_token

    def fetch_lyrics(self, title: str, artist: str) -> LyricsResult:
        if not self._api_token:
            raise RuntimeError(
                "Genius API token not configured. Set GENIUS_API_TOKEN in .env"
            )

        song_url = self._search_song(title, artist)
        lyrics = self._scrape_lyrics(song_url)
        return LyricsResult(title=title, artist=artist, lyrics=lyrics)

    def _search_song(self, title: str, artist: str) -> str:
        query = f"{title} {artist}"
        headers = {"Authorization": f"Bearer {self._api_token}"}

        with httpx.Client() as client:
            try:
                response = client.get(
                    f"{self.BASE_URL}/search",
                    params={"q": query},
                    headers=headers,
                    timeout=15.0,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError:
                raise RuntimeError("Genius API returned an error")
            except httpx.RequestError:
                raise RuntimeError("Failed to connect to Genius API")

        data = response.json()
        hits = data.get("response", {}).get("hits", [])

        if not hits:
            raise LookupError(f"No lyrics found for '{title}' by {artist}")

        artist_lower = artist.lower()
        for hit in hits:
            result = hit.get("result", {})
            primary_artist = (
                result.get("primary_artist", {}).get("name", "").lower()
            )
            if artist_lower in primary_artist or primary_artist in artist_lower:
                return result["url"]

        # Fallback to first result if no exact artist match
        return hits[0]["result"]["url"]

    def _scrape_lyrics(self, url: str) -> str:
        with httpx.Client() as client:
            try:
                response = client.get(url, timeout=15.0, follow_redirects=True)
                response.raise_for_status()
            except httpx.RequestError:
                raise RuntimeError("Failed to fetch lyrics page")

        soup = BeautifulSoup(response.text, "html.parser")
        containers = soup.select('div[data-lyrics-container="true"]')

        if not containers:
            raise LookupError("Lyrics not found on the page")

        parts = []
        for container in containers:
            for br in container.find_all("br"):
                br.replace_with("\n")
            parts.append(container.get_text())

        lyrics = "\n".join(parts).strip()
        lyrics = re.sub(r"\n{3,}", "\n\n", lyrics)
        lyrics = self._clean_lyrics(lyrics)
        return lyrics

    @staticmethod
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
