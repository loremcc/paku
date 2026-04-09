# tests/fixtures/

Real Instagram screenshots for phase gate evaluation and integration testing.

**This directory must be populated manually.** Screenshots are gitignored (contain PII). Only `.gitkeep` and this README are tracked.

## Directory Structure

```
tests/fixtures/
  .gitkeep
  README.md
  anime/          # 30 screenshots — Phase 2 gate set
  urls/           # 17 screenshots — Phase 1 gate set
  (recipe/)       # Planned — Phase 3
```

Screenshots are organized by extractor. Each subdirectory corresponds to a phase gate test set.

## Phase 0 — OCR Baseline (v0.1)

Original 5 root-level fixture images validated Google Cloud Vision OCR output. Gate criterion: >= 15 meaningful characters per screenshot.

**Gate status: PASSED (2026-03-27).** 5/5 fixtures produced usable OCR text.

Run:
```
pytest tests/test_google_vision_engine.py -m integration -s
```

## Phase 1 — URL Extractor (v0.2)

17 screenshots in `urls/` covering 4 extraction tiers:

| Tier | Description | Count | Gate result |
|------|-------------|-------|-------------|
| 1 | Full URL (github.com, arxiv.org, etc.) | 5 | 100% |
| 2-3 | Non-GitHub domains + author/repo reconstruction | 7 | 71.4% |
| 4 | Name-only (routes to review queue) | 5 | 100% |

**Gate status: PASSED (2026-04-01).** Zero false positives.

Known limitation: IMG_5247 multi-line author/repo (mobile GitHub app) falls to Tier 4 — routes correctly to review_queue but with null keywords.

## Phase 2 — Anime Extractor (v0.3)

30 screenshots in `anime/` covering:

- Dark stories with overlay text (low contrast)
- Post captions in Italian and English
- AniList app screenshots (short-circuit path)
- TikTok/Threads cross-platform screenshots
- IG broadcast channels with reaction chrome
- Carousel/multi-title posts (numbered lists, slide markers)
- Feed cards with episode counts and metadata
- All-caps title blocks (release cards)

**Gate status: PASSED (2026-04-09).** 30/30 = 100% auto-accepted (threshold 80%). Previous: 17/30 = 57% before 3 gate-tuning fixes (enhanced ratio, multi-title auto-accept, punctuation cleanup).

## Phase 3 — Recipe Extractor (v0.4, planned)

Will add a `recipes/` subdirectory with >= 10 screenshots. Gate criterion: >= 85% correct ingredient parsing with split qty/unit/name.

## Credentials

Set one of:
- `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json` (env var)
- `google_vision.api_key: <key>` in `config.yaml` (API key fallback)

Free tier: 1,000 image annotations/month.
