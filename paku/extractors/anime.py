from __future__ import annotations

# Extracts anime/manga titles from Instagram screenshot OCR text.
# Priority: AniList-enriched result when ratio >= 0.8; review queue otherwise.
# Structural template: follows url.py conventions (one public extract(), helpers private).

import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from logging import Logger
from typing import Any

import requests

from ..models import AnimeExtractionResult

# --- Platform detection signals ---

_PLATFORM_SIGNALS: dict[str, list[str]] = {
    "anilist_app": ["ADD TO LIST", "AVERAGE SCORE", "MOST POPULAR"],
    "tiktok": ["by author", "Add a comment for", "View more replies"],
    "threads": ["Thread\n", "views\n", "View activity"],
    "ig_broadcast": ["· Moderator", "members", "YESTERDAY"],
    "ig_story": ["Send message...", "See translation >"],
    "ig_feed": ["Add comment...", "Followed by", "Liked by"],
}

# --- Non-anime content signals ---

_DONGHUA_SIGNALS = ["Chinese Web Novel", "Donghua", "BiliBili", "G.CMay", "studio A-CAT"]
_WESTERN_SIGNALS = ["Cartoon Network", "Adult Swim", "Disney", "Nickelodeon"]

# --- Title extraction regexes ---

# Pattern A: explicit label line (handles emoji between label and separator)
# Continuation line is optional — captured UNLESS the next line starts with a metadata keyword
# (Genres, Episodes, Status, etc.) which prevents "Genres: Action..." from being appended.
_META_CONT = r"(?:\n(?!(?:Genres?|Episodes?|Status|(?:Average\s+)?Score|Rating|Type|Format|Studios?|Source|Season|Cast|Duration)\s*:)[^\n]+)?"
_LABEL_RE = re.compile(
    r"[^\w\n]*(?:anime\s*name|•\s*anime|name\s*(?:of\s*(?:the\s*)?anime)?)"
    r"[^::\-–\n]{0,20}[:\-–]+\s*([^\n]{3,120}" + _META_CONT + r")",
    re.IGNORECASE,
)
_LABEL_CALLED_RE = re.compile(
    r"[^\w\n]*(?:the\s+)?anime\s+is\s+called\s+([^\n]{3,120}" + _META_CONT + r")",
    re.IGNORECASE,
)
_LABEL_NAME_IS_RE = re.compile(
    r"[^\w\n]*(?:the\s+)?anime\s+name\s+is\s+([^\n]{1,80})",
    re.IGNORECASE,
)
_LABEL_BARE_ANIME_RE = re.compile(
    r"^\s*anime\s*[:\-–]+\s*([^\n]{3,120})",
    re.IGNORECASE | re.MULTILINE,
)

# Pattern B: quoted title (120 chars to accommodate long light novel titles)
_QUOTED_RE = re.compile(r'"([^"]{3,120})"')
_PLOT_CONTEXT_RE = re.compile(r"\b(?:Plot|Synopsis|Story)\s*:", re.IGNORECASE)

# Pattern C: numbered list (starts with digit+dot, ALL-CAPS or Title-Case title)
_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+([A-Z][^\n]{2,60})$", re.MULTILINE)

# Pattern D: title with year suffix
_YEAR_RE = re.compile(r"^(.{2,80})\s*\(\d{4}\)\s*$")
_STRIP_YEAR_RE = re.compile(r"\s*\(\d{4}\)\s*$")

# Pattern E2: title-case line + parenthesized romaji on next line
_TITLE_ROMAJI_RE = re.compile(
    r"^(.{3,80})\n\(\s*([A-Za-z\s]{3,80})\s*\)",
    re.MULTILINE,
)

# Pattern F: hashtags
_HASHTAG_RE = re.compile(r"#([A-Za-z][A-Za-z0-9]+)")
_CAMEL_RE = re.compile(r"([a-z])([A-Z])")
_HASHTAG_SUFFIX_RE = re.compile(r"(?:anime|manga|series|edit)\s*$", re.IGNORECASE)


def _transform_hashtag(tag: str) -> str:
    """Strip anime/manga/series suffix, then split CamelCase."""
    cleaned = _HASHTAG_SUFFIX_RE.sub("", tag).strip()
    return _CAMEL_RE.sub(r"\1 \2", cleaned).strip()

# Multi-title detection
_MULTI_DATE_RE = re.compile(
    r"^\d+\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(.+)$",
    re.MULTILINE | re.IGNORECASE,
)

# Discussion context
_DISCUSSION_RE = re.compile(
    r"^(?:Is\s+.+\s+the\s+(?:worst|best)|What\s+(?:if|do\s+you)|Would\s+you)",
    re.IGNORECASE,
)
_DISCUSSION_TITLE_RE = re.compile(
    r"^Is\s+(.+?)\s+(?:(?:dub|sub)\s+)?the\s+(?:worst|best)",
    re.IGNORECASE,
)

# Broadcast channel reaction rows
_REACTION_ROW_RE = re.compile(r"^[❤️🔥🌸⚡💯+\d\s]+$")
_TIMESTAMP_LINE_RE = re.compile(r"^YESTERDAY\s+\d+:\d+\s+[AP]M$")
_MODERATOR_LINE_RE = re.compile(r"^.+\s+·\s+Moderator$")

# AniList GraphQL query
_ANILIST_URL = "https://graphql.anilist.co"
_ANILIST_QUERY = """
query ($search: String, $type: MediaType) {
  Media(search: $search, type: $type, sort: SEARCH_MATCH) {
    id
    title { romaji english native }
    type episodes status genres averageScore siteUrl
    coverImage { large }
  }
}
"""


# --- Chrome stripping ---


def _detect_platform(text: str) -> str:
    for platform, signals in _PLATFORM_SIGNALS.items():
        if platform == "anilist_app":
            # Require all three signals — they're very specific
            if all(s in text for s in signals):
                return platform
        else:
            # Any single signal is sufficient for other platforms
            if any(s in text for s in signals):
                return platform
    return "unknown"


def _strip_chrome(text: str, platform: str) -> str:
    lines = text.splitlines()
    kept: list[str] = []

    # Pre-pass: collect lowercase usernames so we can strip their all-caps variants
    # (e.g. "ANIMECULTIVATED" header when "animecultivated" appears as username below)
    _lower_usernames: set[str] = set()
    for line in lines:
        s = line.strip()
        if re.match(r"^[\w.][\w.]{2,30}$", s) and not s.isupper():
            _lower_usernames.add(s.lower())

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue

        # All-caps single-word line matching a detected username elsewhere
        if " " not in stripped and stripped.isupper() and stripped.lower() in _lower_usernames:
            continue

        # Broadcast channel: strip header, moderator, timestamp, reaction rows
        if platform == "ig_broadcast":
            if "· members" in stripped or "· Moderator" in stripped:
                continue
            if _TIMESTAMP_LINE_RE.match(stripped) or _MODERATOR_LINE_RE.match(stripped):
                continue
            if _REACTION_ROW_RE.match(stripped):
                continue

        # Bottom nav / standalone button labels
        if stripped in {"Home", "Inbox", "Explore", "Profile", "Anime", "Manga", "Discover", "Feed", "Likes"}:
            continue

        # Section headers (Reels tab, Explore tab, Comments sheet)
        # Also handles combined tab labels: "Reels Friends", "Explore Friends D"
        if re.match(r"^(?:Comments|For\s+you|(?:Explore|Friends|Reels)(?:\s+\w+)*)\s*[🤪🤝✨]*\s*[Vv>]?\s*$", stripped):
            continue

        # Pure timestamp tokens (3h, 2d, 7w)
        if re.match(r"^\d+[hmwdHMWD]$", stripped):
            continue

        # Username + timestamp (e.g. "syed_akram6143 19h", "conroy_robinson 16w✰")
        if re.match(r"^[\w.][\w.]{1,30}\s+\d+[hmwdHMWD]\b", stripped):
            continue

        # Pure numeric / engagement counts
        if re.match(r"^[\d.,]+[KkMm]?\s*$", stripped):
            continue

        # Engagement lines: "Liked by X and N others", "Followed by X"
        if re.match(r"^(?:Liked\s+by|Followed\s+by)\s+", stripped, re.IGNORECASE):
            continue

        # "Suggested for you" lines
        if re.match(r"^Suggested\s+for\s+you", stripped, re.IGNORECASE):
            continue

        # "Follow" button (standalone or with username on same line)
        if stripped == "Follow":
            continue

        # @username lines (account handles used as attributions / captions)
        if re.match(r"^@\S+", stripped):
            continue

        # Action prompts / input placeholders
        if re.match(r"^(?:Add\s+(?:a\s+)?comment|Reply to|See translation|Send message|Not interested|What do you think)", stripped, re.IGNORECASE):
            continue

        # "View N more replies" (generic, not just TikTok)
        if re.match(r"^View\s+\d+\s+more\s+repl", stripped, re.IGNORECASE):
            continue

        # Swipe indicators
        if re.match(r"^(?:SWIPE|swipe)\s*$", stripped):
            continue

        # Slide counters (e.g. "2/10", "5/12")
        if re.match(r"^\d+/\d+$", stripped):
            continue

        # Engagement stats line (hearts, comments, shares, saves)
        if re.match(r"^[♡❤️🤍💬🔁📤💾\s\d.,KkMm]+$", stripped):
            continue

        # Copyright lines (e.g. ©2026 Author/KADOKAWA...)
        if re.match(r"^©", stripped):
            continue

        # Hashtag-heavy (>50% tokens start with #)
        tokens = stripped.split()
        if tokens and sum(1 for t in tokens if t.startswith("#")) / len(tokens) > 0.5:
            continue

        # Reaction emoji rows (broadcast + comment threads)
        if _REACTION_ROW_RE.match(stripped):
            continue

        # TikTok chrome
        if platform == "tiktok" and (
            stripped in {"Comments", "Reply"}
            or re.match(r"^by\s+\w+", stripped, re.IGNORECASE)
        ):
            continue

        # Username + caption pattern: "username Caption text..." (IG feed captions)
        # Only strip when it looks like a non-anime social caption
        if re.match(r"^[\w.][\w.]{1,30}\s+(?:Follow|Suggested\s+for)", stripped, re.IGNORECASE):
            continue

        kept.append(line)

    return "\n".join(kept)


# --- Multi-title detection ---


_CAROUSEL_MARKER_RE = re.compile(
    r"^(?:ANIME(?:\s+NEWS\s+LIST)?|NEWS\s+LIST|MAL|SEASON\s+\d+|ANISAUCE|SWIPE)$",
    re.IGNORECASE,
)


def _detect_carousel_titles(text: str) -> list[str]:
    """Extract titles from anime news carousel format.

    Walks lines between markers, collecting Title Case segments.
    Everything between two markers (that passes filters) is joined as one title.
    Requires 2+ marker occurrences and 2+ extracted titles to trigger.
    """
    lines = text.splitlines()
    marker_count = sum(1 for l in lines if _CAROUSEL_MARKER_RE.match(l.strip()))
    if marker_count < 2:
        return []

    titles: list[str] = []
    current_parts: list[str] = []

    def _flush() -> None:
        if current_parts:
            titles.append(" ".join(current_parts))
            current_parts.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _CAROUSEL_MARKER_RE.match(stripped):
            _flush()
            continue
        if len(stripped) < 5:
            continue
        alpha = [c for c in stripped if c.isalpha()]
        if not alpha:
            continue
        # Skip all-caps noise (labels, not titles)
        if all(c.isupper() for c in alpha) and len(alpha) > 1:
            continue
        # Skip year lines
        if re.match(r"^\d{4}$", stripped):
            continue
        if _is_garbage_fallback(stripped):
            continue
        # Title Case: first letter uppercase
        if stripped[0].isupper():
            current_parts.append(stripped)

    _flush()
    return titles if len(titles) >= 2 else []


def _detect_multi_titles(text: str) -> list[str]:
    """Return list of title strings from date-prefixed or carousel formats."""
    matches = _MULTI_DATE_RE.findall(text)
    if len(matches) >= 3:
        return [m.strip() for m in matches]

    carousel = _detect_carousel_titles(text)
    if carousel:
        return carousel

    return []


# --- Fallback rejection guard ---

# Username pattern: word chars and dots, 2-30 chars (Instagram/TikTok handles)
_USERNAME_RE = re.compile(r"^[\w.][\w.]{1,30}$")

# Patterns that signal Instagram/social UI noise, not anime titles
_GARBAGE_PATTERNS: list[re.Pattern[str]] = [
    # Username + timestamp ("syed_akram6143 19h", "conroy_robinson 16w✰")
    re.compile(r"^[\w.][\w.]{1,30}\s+\d+[hmwdHMWD]"),
    # "Liked by X and N others"
    re.compile(r"Liked\s+by\s+\S+.*\d+\s+others?", re.IGNORECASE),
    # "Followed by X"
    re.compile(r"^Followed\s+by\s+", re.IGNORECASE),
    # Username + generic social caption ("animelif3 Anime I don't see anyone talking about")
    re.compile(r"^[\w.][\w.]{1,30}\s+(?:Anime\s+I\s+|The\s+best\s+|Donald\s+|I\s+don'?t\s+|Check\s+out\s+|New\s+details?\s+)", re.IGNORECASE),
    # Engagement/community count lines ("COLLECTING ALL ANIME FANS (23k/25k")
    re.compile(r"\(\d+[KkMm]?/\d+[KkMm]?\s*"),
    # "... more" truncated caption
    re.compile(r"^[\w.][\w.]{1,30}\s+.{10,}\.{3}\s*more\s*$", re.IGNORECASE),
    # "by author" TikTok attribution
    re.compile(r"\bby\s+author\b", re.IGNORECASE),
    # Standalone "Name?" or "Reply" (comment thread noise)
    re.compile(r"^(?:Name\??|Reply)\s*$", re.IGNORECASE),
    # "Suggested for you"
    re.compile(r"^Suggested\s+for\s+you", re.IGNORECASE),
    # "Follow" button
    re.compile(r"^Follow\s*$"),
    # "View N more replies"
    re.compile(r"^View\s+\d+\s+more\s+repl", re.IGNORECASE),
    # "Add comment"
    re.compile(r"^Add\s+(?:a\s+)?comment", re.IGNORECASE),
]


def _is_garbage_fallback(text: str) -> bool:
    """Return True if the fallback candidate is social UI noise, not an anime title."""
    return any(p.search(text) for p in _GARBAGE_PATTERNS)


# --- Title + episode count (Pattern I) ---

_TITLE_EPISODE_RE = re.compile(
    r"^(.{3,80})\n\s*\d+\s+Episodes?\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# --- Release card detection (Pattern H) ---

_RELEASE_ANCHORS_RE = re.compile(
    r"^\s*(?:"
    r"(?:RELEASE\s+DATE|STUDIOS?|GENRES?|STREAMING|EPISODES?)\s*:"
    r"|SEASON\s+\d+"
    r"|\d+\s+EPS\b"
    r"|\d+\s+Episodes?\b"
    r")",
    re.IGNORECASE,
)
_RELEASE_NOISE_EXACT = {"ANISAUCE", "MAL", "ANIME NEWS LIST"}
_ALLCAPS_LINE_RE = re.compile(r"^[A-Z0-9][A-Z0-9\s:\-'\[\]]+$")


# --- Title extraction cascade ---


def _extract_title(text: str, full_text: str | None = None) -> tuple[str | None, str | None]:
    """Run pattern cascade A→G. Returns (raw_title, title_pattern)."""

    # Pattern A: explicit label (join continuation line if OCR wrapped the title)
    m = _LABEL_RE.search(text) or _LABEL_CALLED_RE.search(text) or _LABEL_NAME_IS_RE.search(text) or _LABEL_BARE_ANIME_RE.search(text)
    if m:
        return m.group(1).replace("\n", " ").strip(), "label"

    # Pattern B: quoted title (reject conversational placeholders and plot descriptions)
    m = _QUOTED_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        reject = candidate.endswith("?")
        if not reject:
            lines = text.splitlines()
            match_line_idx = text[:m.start()].count("\n")
            for offset in range(-2, 3):
                idx = match_line_idx + offset
                if 0 <= idx < len(lines) and _PLOT_CONTEXT_RE.search(lines[idx]):
                    reject = True
                    break
        if not reject:
            return candidate, "quoted"

    # Pattern C: numbered list
    m = _NUMBERED_RE.search(text)
    if m:
        return m.group(1).strip(), "numbered"

    # Pattern D: title with year suffix
    for line in text.splitlines():
        m = _YEAR_RE.match(line.strip())
        if m and len(m.group(1).strip()) >= 3:
            return m.group(1).strip(), "year_tagged"

    # Pattern E2: title-case line + parenthesized romaji on next line
    m = _TITLE_ROMAJI_RE.search(text)
    if m:
        return m.group(1).strip(), "title_romaji"

    # Pattern E: romaji / CJK (lines with CJK chars or macrons — pass as-is to AniList)
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) < 3:
            continue
        if re.search(r"[一-龯ぁ-んァ-ヶ]", stripped) or re.search(r"[ōūāīē]", stripped, re.IGNORECASE):
            return stripped, "romaji"

    # Pattern F: hashtag (scan full text for 3+ hashtag trigger, not just chrome-stripped)
    hashtag_source = full_text if full_text is not None else text
    hashtags = _HASHTAG_RE.findall(hashtag_source)
    if len(hashtags) >= 3:
        # Pick longest non-generic hashtag
        best = max(
            (h for h in hashtags if len(h) > 4),
            key=len,
            default=None,
        )
        if best:
            title = _transform_hashtag(best)
            if len(title) >= 3:
                return title, "hashtag"

    # Pattern G: discussion title (noun phrase from rhetorical question)
    for line in text.splitlines():
        m = _DISCUSSION_TITLE_RE.match(line.strip())
        if m:
            return m.group(1).strip(), "discussion"

    # Pattern H: all-caps title block + release card anchor (within 2 lines)
    # Joins consecutive all-caps lines, but stops joining when the next line
    # is itself a release anchor (e.g. "26 EPS", "SEASON 3") — that line
    # validates the block above it rather than being part of the title.
    rc_lines = text.splitlines()
    allcaps_candidates: list[str] = []  # saved for Pattern J (no-anchor fallback)
    i = 0
    while i < len(rc_lines):
        line = rc_lines[i].strip()
        if not line or len(line) < 3:
            i += 1
            continue
        alpha_chars = [c for c in line if c.isalpha()]
        if not alpha_chars or not all(c.isupper() for c in alpha_chars) or not _ALLCAPS_LINE_RE.match(line):
            i += 1
            continue
        # Skip lines that are anchors themselves (e.g. "SEASON 2" starting a block)
        if _RELEASE_ANCHORS_RE.match(line):
            i += 1
            continue
        # Join consecutive all-caps lines, stopping at anchors
        parts = [line]
        j = i + 1
        anchor_found = False
        while j < len(rc_lines):
            nxt = rc_lines[j].strip()
            nxt_alpha = [c for c in nxt if c.isalpha()]
            if nxt and nxt_alpha and all(c.isupper() for c in nxt_alpha) and _ALLCAPS_LINE_RE.match(nxt):
                # If this line is an anchor, it validates the block — don't join it
                if _RELEASE_ANCHORS_RE.match(nxt):
                    anchor_found = True
                    j += 1
                    break
                parts.append(nxt)
                j += 1
            else:
                break
        candidate = " ".join(parts)
        # Noise guard
        if candidate in _RELEASE_NOISE_EXACT or re.search(r"\bANIME\b", candidate):
            i = j
            continue
        words = candidate.split()
        if len(words) < 2:
            i = j
            continue
        if anchor_found:
            return candidate, "release_card"
        # Check for anchor within 2 lines after the block
        for offset in range(0, 2):
            idx = j + offset
            if idx < len(rc_lines) and _RELEASE_ANCHORS_RE.match(rc_lines[idx].strip()):
                return candidate, "release_card"
        # No anchor — save for Pattern J
        allcaps_candidates.append(candidate)
        i = j

    # Pattern I: title line followed by "N Episodes" on next line
    m = _TITLE_EPISODE_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        if not _is_garbage_fallback(candidate):
            return candidate, "episode_card"

    # Pattern J: longest all-caps block found in Pattern H that had no anchor
    # Used for meme/comparison posts where titles appear in all-caps without metadata
    if allcaps_candidates:
        best_allcaps = max(allcaps_candidates, key=lambda c: len(c.split()))
        if len(best_allcaps.split()) >= 3:
            return best_allcaps, "allcaps_title"

    # Fallback: longest non-chrome, non-hashtag line — with rejection guard
    candidates = [
        line.strip() for line in text.splitlines()
        if len(line.strip()) >= 3
        and not line.strip().startswith("#")
        and not re.match(r"^\d+[hmwdHMWD]$", line.strip())
    ]
    if candidates:
        best = max(candidates, key=len)
        if _is_garbage_fallback(best):
            return None, None
        return best, "fallback"

    return None, None


# --- AniList query ---


def _query_anilist(
    search: str, media_type: str, logger: Logger
) -> tuple[dict | None, str | None]:
    """POST to AniList GraphQL. Returns (media_dict | None, error_type | None)."""
    try:
        resp = requests.post(
            _ANILIST_URL,
            json={"query": _ANILIST_QUERY, "variables": {"search": search, "type": media_type}},
            timeout=10,
        )
        # AniList returns 404 when no anime matches the search term — treat as no result.
        if resp.status_code == 404:
            logger.debug(f"[anime] AniList no result (404) for '{search}' type={media_type}")
            return None, None
        resp.raise_for_status()
        data = resp.json()
        # Use `or {}` to handle {"data": null, "errors": [...]} responses from AniList.
        media = (data.get("data") or {}).get("Media")
        return media, None
    except Exception as e:
        logger.exception(f"[anime] AniList call failed for '{search}' type={media_type}: {e}")
        return None, "network_error"


def _levenshtein(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _enhanced_ratio(raw: str, canonical: str) -> float:
    """Compute enhanced similarity: base Levenshtein, then substring and word-overlap boosts.

    Returns the boosted ratio (capped at 0.85) when the base ratio is below 0.8
    but structural similarity is high. Returns the base ratio otherwise.
    """
    base = _levenshtein(raw, canonical)
    if base >= 0.8 or not raw or not canonical:
        return base

    raw_l = raw.lower().strip()
    canon_l = canonical.lower().strip()

    # Substring boost: raw is a meaningful prefix/substring of canonical
    if len(raw_l) >= 8 and raw_l in canon_l:
        return max(base, 0.85)

    # Word-set overlap boost: same words in different order (e.g. reordered title)
    # Strip punctuation from word boundaries for clean comparison
    raw_words = {re.sub(r"[^\w-]", "", w) for w in raw_l.split()} - {""}
    canon_words = {re.sub(r"[^\w-]", "", w) for w in canon_l.split()} - {""}
    if len(raw_words) >= 3 and len(canon_words) >= 3:
        overlap = len(raw_words & canon_words) / max(len(raw_words | canon_words), 1)
        if overlap >= 0.75:
            return max(base, 0.85)

    return base


def _compute_best_ratio(raw_title: str, titles: dict) -> float:
    """Compute best enhanced ratio against english and romaji titles."""
    return max(
        _enhanced_ratio(raw_title, titles.get("english") or ""),
        _enhanced_ratio(raw_title, titles.get("romaji") or ""),
    )


def _query_anilist_best(
    query_title: str, raw_title: str, logger: Logger
) -> tuple[dict | None, float, str | None]:
    """Try ANIME then MANGA if ratio < 0.5. Returns (media, ratio, error_type)."""
    # Strip year and leading numbers before querying
    search = _STRIP_YEAR_RE.sub("", query_title).strip()
    search = re.sub(r"^\d+\.\s*", "", search).strip()
    if not search:
        return None, 0.0, None

    media, err = _query_anilist(search, "ANIME", logger)
    if err:
        return None, 0.0, err

    ratio = 0.0
    if media:
        titles = media.get("title", {})
        ratio = _compute_best_ratio(raw_title, titles)
        logger.debug(f"[anime] ANIME ratio={ratio:.2f} for '{search}'")

    if ratio < 0.5:
        manga_media, err = _query_anilist(search, "MANGA", logger)
        if err:
            return media, ratio, err
        if manga_media:
            titles_m = manga_media.get("title", {})
            ratio_m = _compute_best_ratio(raw_title, titles_m)
            logger.debug(f"[anime] MANGA ratio={ratio_m:.2f} for '{search}'")
            if ratio_m > ratio:
                return manga_media, ratio_m, None

    return media, ratio, None


# --- Confidence assignment (single location) ---


def _assign_confidence(
    ratio: float, title_pattern: str | None, extraction_context: str
) -> tuple[float, bool]:
    """Return (confidence, needs_review)."""
    if ratio >= 0.8:
        confidence, needs_review = 0.9, False
    elif ratio >= 0.6:
        confidence, needs_review = 0.7, True
    elif ratio >= 0.4:
        confidence, needs_review = 0.4, True
    else:
        confidence, needs_review = 0.3, True

    if title_pattern == "label":
        confidence = min(confidence + 0.1, 0.95)
    if extraction_context == "discussion":
        confidence -= 0.2

    return confidence, needs_review


# --- AniList app parse path ---


def _parse_anilist_app(
    ocr_text: str, screenshot_path: str, now: str
) -> AnimeExtractionResult:
    lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
    # First long centered line is the title
    raw_title = next(
        (l for l in lines if len(l) >= 3 and not any(s in l for s in ["ADD TO LIST", "AVERAGE SCORE", "MOST POPULAR"])),
        lines[0] if lines else "unknown",
    )
    dedup_key = raw_title.lower().strip()
    return AnimeExtractionResult(
        extractor="anime",
        confidence=0.95,
        needs_review=False,
        source_screenshot=screenshot_path,
        extracted_at=now,
        raw_title=raw_title,
        extraction_mode="anilist_app",
        media_source="anilist_app",
        extraction_context="recommendation",
        dedup_key=dedup_key,
        levenshtein_ratio=None,
    )


# --- Single title processing ---


def _process_single_title(
    raw_title: str,
    title_pattern: str | None,
    extraction_context: str,
    stripped_text: str,
    screenshot_path: str,
    now: str,
    config: dict[str, Any],
    logger: Logger,
    multi_title: bool = False,
    alt_query: str | None = None,
) -> AnimeExtractionResult:
    # Detect platform signals — log but still query AniList (signal only wins if ratio < 0.4)
    signal_type: str | None = None
    donghua_match = next((s for s in _DONGHUA_SIGNALS if s in stripped_text), None)
    if donghua_match:
        logger.debug(f"[anime] donghua signal matched: '{donghua_match}' — querying AniList anyway")
        signal_type = "donghua"
    else:
        western_match = next((s for s in _WESTERN_SIGNALS if s in stripped_text), None)
        if western_match:
            logger.debug(f"[anime] western signal matched: '{western_match}' — querying AniList anyway")
            signal_type = "western"

    # Clean trailing punctuation before AniList query (e.g. "Rage of Bahamut •")
    clean_title = re.sub(r"[\s•·.,;:!?\-–—]+$", "", raw_title).strip()

    # AniList query (always — signal_type is a hint, not a gate)
    media, ratio, err = _query_anilist_best(clean_title, clean_title, logger)

    # Try alternate query (romaji) when primary ratio is low
    if not err and ratio < 0.5 and alt_query:
        alt_media, alt_ratio, alt_err = _query_anilist_best(alt_query, clean_title, logger)
        if not alt_err and alt_ratio > ratio:
            logger.debug(f"[anime] alt_query '{alt_query}' improved ratio {ratio:.2f} -> {alt_ratio:.2f}")
            media, ratio = alt_media, alt_ratio

    if err == "network_error":
        return AnimeExtractionResult(
            extractor="anime", confidence=0.3, needs_review=True,
            source_screenshot=screenshot_path, extracted_at=now,
            raw_title=raw_title, title_pattern=title_pattern,
            media_source=signal_type or "unknown", extraction_context=extraction_context,
            multi_title_detected=multi_title,
            dedup_key=raw_title.lower().strip(), levenshtein_ratio=None,
        )

    confidence, needs_review = _assign_confidence(ratio, title_pattern, extraction_context)

    if multi_title and ratio < 0.8:
        needs_review = True

    # Enrich from AniList result when ratio >= 0.4
    anilist_id: int | None = None
    canonical_title: str | None = None
    native_title: str | None = None
    romaji: str | None = None
    media_type: str | None = None
    media_source = signal_type or "unknown"  # overridden below when AniList ratio >= 0.4
    episodes: int | None = None
    status: str | None = None
    genres: list[str] = []
    score: float | None = None
    anilist_url: str | None = None
    cover_image: str | None = None

    if media and ratio >= 0.4:
        titles = media.get("title", {})
        anilist_id = media.get("id")
        canonical_title = titles.get("english") or titles.get("romaji")
        native_title = titles.get("native")
        romaji = titles.get("romaji")
        media_type = media.get("type")
        media_source = "anime" if media_type == "ANIME" else "manga" if media_type == "MANGA" else "unknown"
        episodes = media.get("episodes")
        status = media.get("status")
        genres = media.get("genres") or []
        raw_score = media.get("averageScore")
        score = raw_score / 10.0 if raw_score is not None else None
        anilist_url = media.get("siteUrl")
        cover_image = (media.get("coverImage") or {}).get("large")

    dedup_key = str(anilist_id) if anilist_id else (canonical_title or raw_title).lower().strip()

    return AnimeExtractionResult(
        extractor="anime",
        confidence=confidence,
        needs_review=needs_review,
        source_screenshot=screenshot_path,
        extracted_at=now,
        raw_title=raw_title,
        canonical_title=canonical_title,
        native_title=native_title,
        romaji=romaji,
        media_type=media_type,
        media_source=media_source,
        episodes=episodes,
        status=status,
        genres=genres,
        score=score,
        anilist_id=anilist_id,
        anilist_url=anilist_url,
        cover_image=cover_image,
        extraction_mode="fast",
        title_pattern=title_pattern,
        extraction_context=extraction_context,
        multi_title_detected=multi_title,
        dedup_key=dedup_key,
        levenshtein_ratio=ratio,  # 0.0 when no AniList match; None only on network error (early return above)
    )


# --- Main entry point ---


def extract(
    ocr_text: str,
    screenshot_path: str,
    config: dict[str, Any],
    logger: Logger,
) -> AnimeExtractionResult | list[AnimeExtractionResult]:
    """Extract anime title(s) from OCR text. Returns list only for multi-title screenshots."""
    now = datetime.now(timezone.utc).isoformat()

    # Step 0: platform detection + AniList app short-circuit
    platform = _detect_platform(ocr_text)
    if platform == "anilist_app":
        logger.debug("[anime] AniList app detected — short-circuit path")
        return _parse_anilist_app(ocr_text, screenshot_path, now)

    stripped = _strip_chrome(ocr_text, platform)
    logger.debug(f"[anime] chrome stripped: {len(ocr_text)} -> {len(stripped)} chars")

    # Step 1: multi-title detection
    multi_titles = _detect_multi_titles(stripped)
    if multi_titles:
        logger.debug(f"[anime] multi-title: {len(multi_titles)} titles")
        results = []
        for raw_title in multi_titles:
            context = "discussion" if _DISCUSSION_RE.match(raw_title) else "recommendation"
            res = _process_single_title(
                raw_title=raw_title, title_pattern="label",
                extraction_context=context, stripped_text=stripped,
                screenshot_path=screenshot_path, now=now,
                config=config, logger=logger, multi_title=True,
            )
            results.append(res)
        return results

    # Step 2: single title extraction cascade (pass original text for Pattern F hashtag scan)
    raw_title, title_pattern = _extract_title(stripped, full_text=ocr_text)
    if raw_title is None:
        logger.debug("[anime] no extractable title — all patterns failed or fallback rejected garbage")
        return AnimeExtractionResult(
            extractor="anime",
            confidence=0.0,
            needs_review=True,
            source_screenshot=screenshot_path,
            extracted_at=now,
            raw_title="",
            title_pattern=None,
            extraction_context="recommendation",
            media_source="unknown",
            dedup_key=f"no_title_{screenshot_path}",
            levenshtein_ratio=None,
        )

    # Step 2b: extract romaji alternate query for title_romaji pattern
    alt_query: str | None = None
    if title_pattern == "title_romaji":
        m = _TITLE_ROMAJI_RE.search(stripped)
        if m:
            alt_query = m.group(2).strip()

    # Step 3: discussion context detection
    extraction_context = "discussion" if (_DISCUSSION_RE.match(stripped.lstrip()) or title_pattern == "discussion") else "recommendation"

    logger.debug(f"[anime] extracted title='{raw_title}' pattern={title_pattern} context={extraction_context}")

    # Preserve extracted title before entering AniList call path — if an
    # exception escapes _query_anilist's try/except, this value survives.
    extracted_raw_title = raw_title

    try:
        return _process_single_title(
            raw_title=raw_title, title_pattern=title_pattern,
            extraction_context=extraction_context, stripped_text=stripped,
            screenshot_path=screenshot_path, now=now,
            config=config, logger=logger, multi_title=False,
            alt_query=alt_query,
        )
    except Exception as e:
        logger.exception(f"[anime] AniList call failed, preserving extracted title '{extracted_raw_title}': {e}")
        return AnimeExtractionResult(
            extractor="anime", confidence=0.3, needs_review=True,
            source_screenshot=screenshot_path, extracted_at=now,
            raw_title=extracted_raw_title,
            title_pattern=title_pattern,
            media_source="unknown",
            extraction_context=extraction_context,
            dedup_key=extracted_raw_title.lower().strip(),
            levenshtein_ratio=None,
        )
