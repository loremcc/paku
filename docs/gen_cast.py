"""Generate docs/demo.cast in asciinema v2 NDJSON format.

Programmatic generation because asciinema requires fcntl (Linux-only).
Run: python docs/gen_cast.py
Then: tools/agg.exe docs/demo.cast docs/demo.gif --font-size 14 --speed 1.5
"""
from __future__ import annotations

import json
from pathlib import Path

WIDTH = 100
HEIGHT = 30
PROMPT = "$ "

# Actual paku output for IMG_5417.PNG (captured 2026-04-27)
PAKU_OUTPUT_LINES = [
    "--- IMG_5417.PNG ---",
    "  screen_type : post",
    "  content_type: anime",
    "  engine      : google_vision",
    "  url         : —",
    "  confidence  : 0.9",
    "  tier        : —",
    "  ocr_text    :",
    "    01:40",
    "    ANIMECULTIVATED",
    "    RAINBOW: NISHA ROKUBŌ NO SHICHININ",
    "    Completed",
    "    6/12",
    "    26 EPS",
    "    -",
    "    A raw, emotional, painfully",
    "    human story - the kind",
    "    MAL 8.46|2010",
    "    animecultivated",
]

# Full JSON content from output/IMG_5417.json
JSON_LINES = [
    "{",
    '  "extractor": "anime",',
    '  "confidence": 0.9,',
    '  "needs_review": false,',
    '  "raw_title": "RAINBOW: NISHA ROKUBŌ NO SHICHININ",',
    '  "canonical_title": "Rainbow",',
    '  "native_title": "RAINBOW -二舎六房の七人-",',
    '  "romaji": "RAINBOW: Nisha Rokubou no Shichinin",',
    '  "media_type": "ANIME",',
    '  "media_format": "TV",',
    '  "episodes": 26,',
    '  "status": "FINISHED",',
    '  "genres": ["Drama", "Psychological", "Thriller"],',
    '  "score": 8.2,',
    '  "studios": ["MADHOUSE"],',
    '  "country_of_origin": "Japan",',
    '  "debut_year": 2010,',
    '  "anilist_id": 6114,',
    '  "anilist_url": "https://anilist.co/anime/6114",',
    '  "levenshtein_ratio": 0.9565217391304348',
    "}",
]


def build_events() -> list[list]:
    events: list[list] = []
    t = 0.0

    def emit(text: str, dt: float = 0.0) -> None:
        nonlocal t
        t += dt
        events.append([round(t, 3), "o", text])

    # Opening pause
    t = 0.4

    # Prompt + typed command
    emit(PROMPT)
    cmd = "paku digest tests/fixtures/anime/IMG_5417.PNG --mode anime --output json"
    chunk_size = 4
    for i in range(0, len(cmd), chunk_size):
        emit(cmd[i : i + chunk_size], 0.045)
    emit("\r\n", 0.12)

    # Simulate OCR round-trip
    t += 1.9

    # paku output
    for i, line in enumerate(PAKU_OUTPUT_LINES):
        dt = 0.02 if i < 7 else 0.05
        emit(line + "\r\n", dt)

    # Pause before second command
    t += 0.5
    emit(PROMPT)

    cmd2 = "cat output/IMG_5417.json"
    for i in range(0, len(cmd2), chunk_size):
        emit(cmd2[i : i + chunk_size], 0.05)
    emit("\r\n", 0.12)

    t += 0.2

    # JSON output
    for line in JSON_LINES:
        emit(line + "\r\n", 0.035)

    # Final prompt
    t += 0.4
    emit(PROMPT)

    return events


def main() -> None:
    out = Path(__file__).parent / "demo.cast"
    header = {
        "version": 2,
        "width": WIDTH,
        "height": HEIGHT,
        "timestamp": 1745776000,
        "title": "paku demo — anime extraction + AniList enrichment",
        "env": {"TERM": "xterm-256color", "SHELL": "/bin/bash"},
    }
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for evt in build_events():
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")
    print(f"Written: {out}  ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
