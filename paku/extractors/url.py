from __future__ import annotations

import re
from datetime import datetime, timezone
from logging import Logger
from typing import Any

from ..models import URLExtractionResult
from ..pipeline import DOMAIN_PATTERNS

# --- Shared constants ---

SOCIAL_DOMAIN_BLOCKLIST = frozenset({
    "instagram.com",
    "threads.net",
    "facebook.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
    "youtube.com",
    "reddit.com",
})

# Curated TLD allowlist (~60 TLDs).  Prevents matching file extensions (.py,
# .json, .png), version strings (v2.py), and abbreviations (e.g, i.e).
TLD_ALLOWLIST = frozenset({
    "com", "org", "net", "io", "dev", "co", "app", "ai", "xyz", "tech",
    "cloud", "site", "online", "info", "me", "cc", "gg", "tv", "fm", "sh",
    "so", "to", "is", "it", "de", "fr", "uk", "us", "eu", "ru", "jp", "kr",
    "br", "in", "nl", "se", "ch", "at", "be", "au", "ca", "pl", "cz", "fi",
    "no", "dk", "pt", "es", "ro", "hu", "sk", "bg", "hr", "lt", "lv", "ee",
    "ie", "lu",
})

# GitHub context signals — when 2+ are present in OCR text, the screenshot
# is a GitHub repo page and Tier 2 domain-only matches should defer to Tier 3.
GITHUB_CONTEXT_SIGNALS = [
    "Stars", "Forks", "Issues", "Pull requests", "Pull Requests",
    "README", "MIT license", "Contributors", "Discussions", "Actions",
    "Releases", "Code", "master", "main", "commits",
]

# Chrome adjacency patterns — if any appear on the same or adjacent line,
# the candidate is likely an Instagram username, not a target URL.
_CHROME_ADJACENCY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b\d+[hdmw]\b"),                         # "3h", "2d"
    re.compile(r"\d+\s*(?:days?|hours?|minutes?|weeks?)\s*ago", re.IGNORECASE),
    re.compile(r"\bFollow\b"),
    re.compile(r"\bFollowing\b"),
    re.compile(r"\bSuggested for you\b", re.IGNORECASE),
    re.compile(r"\bSend message\b", re.IGNORECASE),
    re.compile(r"\bMessage\b"),
]

# Instagram bottom nav items to strip during noise filtering.
_BOTTOM_NAV = frozenset({"Home", "Inbox", "Explore", "Profile"})

# Engagement line prefixes.
_ENGAGEMENT_PREFIXES = ("Liked by", "Followed by")

# Action prompt prefixes.
_ACTION_PROMPTS = ("Add comment", "Reply to", "See translation")

# Build the TLD alternation for the Tier 2 regex once at import time.
_TLD_ALT = "|".join(sorted(TLD_ALLOWLIST, key=len, reverse=True))
_TIER2_DOMAIN_RE = re.compile(
    rf"\b([a-zA-Z0-9][-a-zA-Z0-9]+\.(?:{_TLD_ALT})(?:/[^\s,)\]\"'<>]*)?)\b"
)

# Tier 1 general URL regex.
_GENERAL_URL_RE = re.compile(r"https?://[^\s,)\]\"'<>]+")

# Tier 3 author/repo regex.
_AUTHOR_REPO_RE = re.compile(r"([\w.-]{1,39})\s*/\s*([\w.-]{1,100})")

# Trailing punctuation to strip from matched URLs.
_TRAILING_PUNCT = set(".,)]\"\\'>;")


# --- Noise stripping ---


def strip_noise(ocr_text: str) -> str:
    """Remove non-content lines from OCR text before extraction.

    Filters (conservative — if a line doesn't clearly match, keep it):
    - Push notifications (first 1-3 lines with notification-like patterns)
    - Reel headers ("For you" + "Friends")
    - Pure numeric lines
    - Engagement lines ("Liked by", "Followed by")
    - Action prompts ("Add comment", "Reply to", "See translation")
    - Hashtag-heavy lines (>50% tokens start with #)
    - Slide indicators (e.g. "1/7")
    - Bottom nav items ("Home", "Inbox", "Explore", "Profile")
    """
    lines = ocr_text.splitlines()
    kept: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue

        # Push notifications — only check first 3 lines.
        # Tightened: require app-name-like start + "now" or timestamp pattern.
        if i < 3 and _is_notification_line(stripped):
            continue

        # Reel headers: line containing both "For you" AND "Friends".
        if "for you" in stripped.lower() and "friends" in stripped.lower():
            continue

        # Pure numeric (engagement counts).
        if re.match(r"^\s*[\d.,]+[KkMm]?\s*$", stripped):
            continue

        # Engagement lines.
        if any(stripped.startswith(p) for p in _ENGAGEMENT_PREFIXES):
            continue

        # Action prompts.
        if any(stripped.startswith(p) for p in _ACTION_PROMPTS):
            continue

        # Hashtag-heavy (>50% of tokens start with #).
        tokens = stripped.split()
        if tokens and sum(1 for t in tokens if t.startswith("#")) / len(tokens) > 0.5:
            continue

        # Slide indicators.
        if re.match(r"^\s*\d+\s*/\s*\d+\s*$", stripped):
            continue

        # Bottom nav.
        if stripped in _BOTTOM_NAV:
            continue

        kept.append(line)

    return "\n".join(kept)


def _is_notification_line(line: str) -> bool:
    """Check if a line looks like a push notification overlay.

    Requires the line to start with an app-name-like word AND contain
    a temporal token ("now", "ago", a short timestamp like "3m", "2h").
    """
    if re.match(r"^[A-Z][a-zA-Z]+\b", line) is None:
        return False
    if re.search(r"\bnow\b", line, re.IGNORECASE):
        return True
    if re.search(r"\b\d+[mhd]\b", line):
        return True
    if re.search(r"\bago\b", line, re.IGNORECASE):
        return True
    return False


# --- Tier implementations ---


def _clean_url(raw: str) -> tuple[str, bool]:
    """Clean a raw URL match. Returns (cleaned_url, was_truncated)."""
    url = raw.strip()
    truncated = False

    # Strip trailing ellipsis.
    if url.endswith("...") or url.endswith("\u2026"):
        truncated = True
        url = url.rstrip(".").rstrip("\u2026")

    # Strip trailing punctuation.
    while url and url[-1] in _TRAILING_PUNCT:
        url = url[:-1]

    # Strip .git suffix.
    if url.endswith(".git"):
        url = url[:-4]

    return url, truncated


def _is_social_domain(url: str) -> bool:
    """Check if URL belongs to a social platform on the blocklist."""
    # Extract domain from URL (with or without scheme).
    m = re.search(r"(?:https?://)?(?:www\.)?([^/]+)", url)
    if not m:
        return False
    domain = m.group(1).lower()
    return any(domain == d or domain.endswith("." + d) for d in SOCIAL_DOMAIN_BLOCKLIST)


def _snippet(text: str, start: int, end: int, context: int = 100) -> str:
    """Extract a snippet around a match position."""
    s = max(0, start - context)
    e = min(len(text), end + context)
    return text[s:e]


def _count_github_signals(text: str) -> int:
    """Count how many GitHub context signals appear in the text."""
    return sum(1 for sig in GITHUB_CONTEXT_SIGNALS if sig in text)


def _has_chrome_adjacency(lines: list[str], line_idx: int) -> bool:
    """Check if chrome signals appear on the same or adjacent lines."""
    check_range = range(
        max(0, line_idx - 1),
        min(len(lines), line_idx + 2),
    )
    for idx in check_range:
        for pat in _CHROME_ADJACENCY_PATTERNS:
            if pat.search(lines[idx]):
                return True
    return False


def _tier1(cleaned_text: str) -> URLExtractionResult | None:
    """Tier 1: Full URL in OCR text (confidence 0.9).

    Two regex passes: general URL, then domain patterns.
    Post-match: social blocklist, truncation detection.
    """
    candidates: list[tuple[str, int, int]] = []

    # Pass 1: general URL regex.
    for m in _GENERAL_URL_RE.finditer(cleaned_text):
        candidates.append((m.group(), m.start(), m.end()))

    # Pass 2: domain patterns (without scheme).
    for pattern in DOMAIN_PATTERNS:
        for m in re.finditer(pattern, cleaned_text):
            candidates.append((m.group(), m.start(), m.end()))

    for raw, start, end in candidates:
        # Check for git clone prefix on the line containing the match.
        line_start = cleaned_text.rfind("\n", 0, start) + 1
        line = cleaned_text[line_start:start]
        if re.search(r"git\s+clone\s+$", line):
            pass  # prefix is in the preceding text, not in raw — no stripping needed

        raw_has_scheme = raw.startswith("http://") or raw.startswith("https://")

        url, truncated = _clean_url(raw)
        if not url:
            continue

        # Prepend scheme if missing.
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url

        # Social domain blocklist.
        if _is_social_domain(url):
            continue

        # Truncation detection: fall through to Tier 3 if the final path segment
        # looks truncated, allowing Tier 3 to reconstruct the full URL.
        # Two paths:
        # 1. Explicit truncation marker (... / …) AND segment < 6 chars.
        # 2. No explicit marker, bare domain (no https:// scheme in source), segment < 6 chars,
        #    AND 2+ GitHub context signals. Browser bar URLs are always bare (no scheme);
        #    a full https:// URL in post text with a short path (e.g. /go, /rs) is legitimate.
        # Why 6: shortest common repo name segment that is likely complete.
        path_part = url.split("/")[-1]
        if truncated and len(path_part) < 6:
            continue
        if not truncated and not raw_has_scheme and len(path_part) < 6 and _count_github_signals(cleaned_text) >= 2:
            # Short final segment on a bare-domain GitHub URL without explicit ellipsis —
            # likely browser bar truncation without a visible marker.
            # Fall through to Tier 3 which may reconstruct the full URL.
            continue

        # Handle hyphen-broken URLs split across two comment lines.
        # e.g. "https://github.com/Anil-matcha/Open-\nHighsfield-Al"
        if url.endswith("-"):
            line_end = cleaned_text.find("\n", end)
            if line_end != -1:
                next_start = line_end + 1
                next_line_end = cleaned_text.find("\n", next_start)
                if next_line_end == -1:
                    next_line_end = len(cleaned_text)
                next_line = cleaned_text[next_start:next_line_end].strip()
                cont = re.match(r"^([\w-]+)", next_line)
                if cont:
                    url = url + cont.group(1)

        snip = _snippet(cleaned_text, start, end)
        return URLExtractionResult(
            confidence=0.9,
            needs_review=False,
            source_screenshot="",  # filled by caller
            extracted_at="",       # filled by caller
            resolved_url=url,
            raw_text_snippet=snip,
            extraction_tier=1,
        )

    return None


def _tier2(cleaned_text: str) -> URLExtractionResult | None:
    """Tier 2: Non-GitHub domain detection (confidence 0.7, needs_review).

    Uses TLD allowlist. Excluded by social blocklist, chrome adjacency,
    and GitHub context suppression (2+ signals → skip entirely to Tier 3).
    """
    # GitHub context suppression: if 2+ signals AND we're actually on a GitHub
    # page (github.com in OCR), skip Tier 2 entirely and let Tier 3 reconstruct
    # the author/repo. Without github.com present, signals come from post text
    # describing a repo (Threads/Instagram feed) — Tier 2 should still fire.
    if _count_github_signals(cleaned_text) >= 2 and "github.com" in cleaned_text.lower():
        return None

    lines = cleaned_text.splitlines()

    for line_idx, line in enumerate(lines):
        for m in _TIER2_DOMAIN_RE.finditer(line):
            candidate = m.group(1)

            # Reject deep path URLs (raw content/asset URLs, not project pages).
            # Allow at most domain + 2 path segments (e.g. example.com/a/b).
            # URLs with 3+ path segments are almost always raw file references,
            # not the project page the screenshot is about.
            if len(candidate.split("/")) > 3:
                continue

            # Social blocklist.
            candidate_with_scheme = candidate
            if not candidate.startswith("http"):
                candidate_with_scheme = "https://" + candidate
            if _is_social_domain(candidate_with_scheme):
                continue

            # Chrome adjacency exclusion.
            if _has_chrome_adjacency(lines, line_idx):
                continue

            # Clean trailing punctuation.
            url = candidate
            while url and url[-1] in _TRAILING_PUNCT:
                url = url[:-1]
            if not url:
                continue

            if not url.startswith("http"):
                url = "https://" + url

            abs_start = sum(len(lines[j]) + 1 for j in range(line_idx)) + m.start()
            snip = _snippet(cleaned_text, abs_start, abs_start + len(candidate))

            return URLExtractionResult(
                confidence=0.7,
                needs_review=True,
                source_screenshot="",
                extracted_at="",
                resolved_url=url,
                raw_text_snippet=snip,
                extraction_tier=2,
            )

    return None


def _tier3(cleaned_text: str) -> URLExtractionResult | None:
    """Tier 3: GitHub author/repo reconstruction (confidence 0.75, needs_review).

    Looks for author/repo pattern near 2+ GitHub context signals.
    Rejects pure numeric groups and slide indicators.
    """
    for m in _AUTHOR_REPO_RE.finditer(cleaned_text):
        author = m.group(1)
        repo = m.group(2)

        # Reject pure numeric.
        if author.isdigit() or repo.isdigit():
            continue

        # Reject domain fragments — GitHub usernames cannot contain dots.
        if "." in author:
            continue

        # Reject path segments inside a longer URL.
        # If the character immediately before the match is '/', the pattern is
        # a subpath (e.g. /master/assets/logo), not a standalone author/repo.
        if m.start() > 0 and cleaned_text[m.start() - 1] == "/":
            continue

        # Reject slide indicators.
        full_match_line = cleaned_text[
            max(0, cleaned_text.rfind("\n", 0, m.start()) + 1):
            cleaned_text.find("\n", m.end())
            if cleaned_text.find("\n", m.end()) != -1
            else len(cleaned_text)
        ].strip()
        if re.match(r"^\s*\d+\s*/\s*\d+\s*$", full_match_line):
            continue

        # Context validation: 500-char window around match, need 2+ signals.
        # Wider window (500 each side) ensures signals visible elsewhere on the
        # page (e.g. MIT license below a header) are counted correctly.
        window_start = max(0, m.start() - 500)
        window_end = min(len(cleaned_text), m.end() + 500)
        window = cleaned_text[window_start:window_end]

        if _count_github_signals(window) < 2:
            continue

        url = f"https://github.com/{author}/{repo}"
        # Strip trailing dots from repo if any.
        url = url.rstrip(".")

        snip = _snippet(cleaned_text, m.start(), m.end())

        return URLExtractionResult(
            confidence=0.75,
            needs_review=True,
            source_screenshot="",
            extracted_at="",
            resolved_url=url,
            raw_text_snippet=snip,
            extraction_tier=3,
        )

    return None


def _extract_project_name(cleaned_text: str) -> str | None:
    """Tier 4 heuristic: extract the most prominent project name.

    - ALL-CAPS words >= 3 chars (excluding common words and IG nav items)
    - Longest capitalized phrase within 2 lines of signal phrases
    """
    signal_phrases = [
        "open source", "open-source", "library", "framework", "tool",
        "project", "repository", "repo",
    ]
    excluded_caps = frozenset({
        "HOME", "INBOX", "EXPLORE", "PROFILE", "THE", "AND", "FOR",
        "YOU", "THIS", "THAT", "WITH", "FROM", "YOUR", "HAS", "ARE",
        "NOT", "BUT", "ALL", "WAS", "CAN",
    })

    lines = cleaned_text.splitlines()
    best: str | None = None
    best_len = 0

    # Strategy 1: longest capitalized phrase near signal phrases.
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if not any(sp in line_lower for sp in signal_phrases):
            continue
        # Check this line and ±2 lines for capitalized phrases.
        for j in range(max(0, i - 2), min(len(lines), i + 3)):
            # Find sequences of capitalized words.
            for m in re.finditer(r"(?:[A-Z][a-zA-Z0-9]*(?:\s+|$)){2,}", lines[j]):
                phrase = m.group().strip()
                if len(phrase) > best_len:
                    best = phrase
                    best_len = len(phrase)

    # Strategy 2: ALL-CAPS words >= 3 chars.
    if not best:
        for m in re.finditer(r"\b[A-Z]{3,}\b", cleaned_text):
            word = m.group()
            if word not in excluded_caps and len(word) > best_len:
                best = word
                best_len = len(word)

    return best


def _tier4(cleaned_text: str) -> URLExtractionResult:
    """Tier 4: Name-only stub (confidence 0.0, needs_review).

    Always returns a result — this is the fallback tier.
    """
    name = _extract_project_name(cleaned_text)
    snip = cleaned_text[-200:] if len(cleaned_text) > 200 else cleaned_text

    return URLExtractionResult(
        confidence=0.0,
        needs_review=True,
        source_screenshot="",
        extracted_at="",
        resolved_url=None,
        raw_text_snippet=snip,
        raw_keywords=name,
        extraction_tier=4,
    )


# --- Main entry point ---


def extract(
    ocr_text: str,
    screenshot_path: str,
    config: dict[str, Any],
    logger: Logger,
) -> URLExtractionResult:
    """Run URL extraction on OCR text using the 4-tier cascade.

    Always returns a URLExtractionResult (Tier 4 is the guaranteed fallback).
    """
    now = datetime.now(timezone.utc).isoformat()

    cleaned = strip_noise(ocr_text)
    logger.debug(f"[url_extractor] noise stripped: {len(ocr_text)} -> {len(cleaned)} chars")

    # Tier 1 — Full URL.
    result = _tier1(cleaned)
    if result is not None:
        logger.debug(f"[url_extractor] Tier 1 match: {result.resolved_url}")
        result.source_screenshot = screenshot_path
        result.extracted_at = now
        return result

    # Tier 2 — Domain-only.
    result = _tier2(cleaned)
    if result is not None:
        logger.debug(f"[url_extractor] Tier 2 match: {result.resolved_url}")
        result.source_screenshot = screenshot_path
        result.extracted_at = now
        return result

    # Tier 3 — GitHub author/repo reconstruction.
    result = _tier3(cleaned)
    if result is not None:
        logger.debug(f"[url_extractor] Tier 3 match: {result.resolved_url}")
        result.source_screenshot = screenshot_path
        result.extracted_at = now
        return result

    # Tier 4 — Name-only fallback (always returns).
    result = _tier4(cleaned)
    logger.debug(f"[url_extractor] Tier 4 fallback: keywords={result.raw_keywords}")
    result.source_screenshot = screenshot_path
    result.extracted_at = now
    return result
