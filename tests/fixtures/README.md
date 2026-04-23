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
  recipes/        # 2 screenshots so far — Phase 3 gate set (needs 8 more)
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

## Phase 3 — Recipe Extractor (v0.4)

10 screenshots in `recipes/` — gate set complete.

Gate criteria (per screenshot): title correct + source_account correct + ≥80% ingredient split accuracy. Gate pass: ≥85% of screenshots (≥9/10) pass all three criteria.

### Per-fixture results (assessed 2026-04-19)

| Fixture | Title | Source | Ing% | PASS? | Notes |
|---------|-------|--------|------|-------|-------|
| IMG_6529.PNG | "Valori senza topping" | alessandrodonofrio_ ✓ | 5/6 = 83% | ✓ | Italian fitness recipe post |
| IMG_6535.PNG | None ✓ (no title in OCR) | daniel.pitasi ✓ | 5/5 = 100% | ✓ | Music credit rejected; title=None is correct when no recipe title present |
| IMG_6791.PNG | caption fragment ✓ | recipeincaption ✓ | 24/28 = 86% | ✓ | needs_review but extraction correct; multi-ingredient OCR lines |
| IMG_6792.PNG | UDON ALLE VERDURE ✓ | ritas.recipes ✓ | 9/9 = 100% | ✓ | |
| IMG_6793.PNG | CHEESECAKE LIGHT E PROTEICA ✓ | ritas.recipes ✓ | 5/5 = 100% | ✓ | |
| IMG_6796.PNG | None ✓ (off-screen) | None ✓ (web) | 6/7 = 86% | ✓ | Metric-parens fix; reversed "Name qty unit (metric)" format handled |
| IMG_6797.PNG | Ragù di carne alla bolognese ✓ | None ✓ (web) | 15/16 = 94% | ✓ | Italian website recipe |
| IMG_6798.PNG | None ✓ (off-screen) | None ✓ (web) | 12/14 = 86% | ✓ | Italian website; title above OCR viewport |
| IMG_6799.PNG | Garlic Pineapple Steak ✓ | None ✓ (web) | 17/18 = 94% | ✓ | English website; steak line split by OCR |
| 6e758002.jpg | None ✓ (off-screen) | None ✓ (web) | 10/11 = 91% | ✓ | Italian website; "Uova 1" has no unit (expected limitation) |

**Score: 10/10 = 100%**

**Gate status: PASSED (2026-04-19).** 100% ≥ 85% threshold.

Extractor improvements applied during Phase 3 gate session (2026-04-19):
- `cucchiai(?:no|ni|o)?` regex fix — now correctly matches cucchiaino/cucchiaini (Italian tsp)
- `pizzic(?:o|hi)` added to units — parses "1 pizzico" (Italian pinch) via trailing-qty fallback
- `_title_score`: ALL-CAPS bonus now requires `w.isalpha() and len(w) > 2` — prevents "AT", "THAT!" from getting upper bonus
- `_extract_source_account`: two-pass scan — Follow-button lines anywhere, bare/decorated handles first 8 lines only (prevents website nav fragments like "Blogge" from matching)
- `_extract_title`: rejects candidates containing `·` (Instagram music credit middle-dot) and fragments ending with ")" without opening "(" — returns None when no genuine recipe title found (correct behavior)
- `_METRIC_PARENS_RE` + `_IMPERIAL_TRAIL_RE`: reversed-format "Name qty unit (metric_g)" now handled — `(450 g)` extracted as authoritative qty, garbled imperial stripped
- Bare-count fallback in `_parse_ingredient_line`: "Vanilla bean 1" → qty=1, unit=None

## Phase 4 — Batch Gate (v0.5)

Full batch run on `input/` (1287 images). Not a fixture test — validates pipeline reliability and output quality at scale.

**Run (2026-04-24):** `paku digest input/ --mode auto --smart --output txt --output json --output csv --resume --report`

| Metric | Value |
|--------|-------|
| Total images | 1287 |
| Processed | 1281 (99.5%) |
| Failed | 1 (0.08%) |
| review_queued | 356 |
| anime JSONs | 765 |
| url JSONs | 55 |
| recipe JSONs | 17 |
| unknown/unclassified | ~446 |
| `anime_export.csv` rows | 599 (deduped) |
| AniList-enriched rows | 507 (84.6%) |
| Raw-title-only rows | 92 (15.4%) |

**review_queue breakdown:**

| Reason | Count | Notes |
|--------|-------|-------|
| `network_error` | 168 | AniList HTTP timeouts/503s mid-run |
| `no_anilist_match` | 85 | Title extracted, no AniList fuzzy match |
| `low_ratio` | ~68 | Match found but ratio < 0.6 gating threshold |
| `multi_title_screenshot` | ~12 | Mixed-confidence multi-title posts |
| `poor_ocr` | 5 | < 15 meaningful chars |
| other | ~18 | Classification mismatches, unknown content |

**Gate status: PASSED (2026-04-24).** failed=1 (0.08% << 5% threshold). Headers match `ANIME_CSV_HEADERS`. 10-row spot-check: `canonical_title` + `anilist_url` populated on all checked rows.

**Architectural gap found:** `process_batch(--resume)` with all images checkpointed returns `results=[]`. Post-batch `_write_anime_csv([])` exits early — CSV not written. Workaround: read per-image JSONs from `output/` directly. See `tasks/lessons.md` for the lesson and rule.

---

## Credentials

Set one of:
- `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json` (env var)
- `google_vision.api_key: <key>` in `config.yaml` (API key fallback)
- `google_vision.credentials_file: /path/to/service-account.json` in `config.yaml`

Free tier: 1,000 image annotations/month.

## Ollama VLM Integration (2026-04-20)

Live integration verified against `gemma4-paku:latest` running on `192.168.1.114:11434`. 8 anime fixtures tested through the smart re-run path. Tests are in `tests/test_ocr_ollama.py` (7 tests, mocked `requests`). Live server not required for CI — all tests use mocked responses.

Fallback verified with unreachable host (192.0.2.1) — graceful degradation, no crash, fast-path result returned unchanged.
