# paku

CLI tool that turns Instagram screenshots into structured data. Built to process a personal backlog of ~3000 screenshots — extracting anime titles, GitHub URLs, and recipes into JSON, CSV, and Notion.

## What it does

You feed it an Instagram screenshot. It runs OCR (Google Cloud Vision), figures out what kind of content it's looking at (anime recommendation, GitHub link, recipe), extracts the relevant data, and writes it out in a usable format.

Three extractors, each purpose-built:

- **URL** (complete, v0.2) — 4-tier extraction cascade validated on 17 real screenshots. Regex-matches full URLs (github.com, arxiv.org, etc.), detects non-GitHub domains via curated TLD allowlist, reconstructs GitHub `author/repo` from repo cards, and stubs project-name-only cases for manual review. Handles browser bar truncation (with and without OCR-visible ellipsis), filters social platform URLs, strips noise, and routes uncertain results to the review queue. Phase 1 gate passed: Tier 1 100%, Tier 2-3 71.4%, Tier 4 100%, zero false positives.
- **Anime** (implemented, v0.3 — gate pending) — 10-pattern title extraction cascade with AniList GraphQL enrichment. Strips Instagram UI chrome (15+ filter categories), detects platform context (AniList app, TikTok, Threads), handles multi-title posts (carousels, numbered lists). Levenshtein ratio gates auto-acceptance (>= 0.8) vs review queue. Currently at 57% auto-accept (80% gate threshold) — analysis of 13 queued screenshots pending.
- **Recipe** (planned, v0.4) — detects ingredient blocks, splits each line into quantity + unit + name (never stored as "100g" — always `{qty: 100, unit: "g"}`).

Anything the pipeline isn't confident about lands in `review_queue.json` instead of being silently discarded.

## Current state

**v0.3 — Anime extractor (implemented, gate pending)**

Both URL and anime extractors are implemented. The pipeline runs end-to-end for URL and anime content. 262 tests pass (22 skipped for missing SDK/credentials). Phase 1 gate passed (2026-04-01). Phase 2 gate NOT YET PASSED — 17/30 = 57% auto-accepted, threshold is 80%. 13 screenshots routed to review_queue need analysis to determine fixable vs. legitimate review cases.

Next up: Phase 2 gate analysis and tuning.

## Install

```bash
git clone https://github.com/loremcc/paku.git
cd paku
pip install -e ".[dev]"
```

For real OCR (not the stub engine):

```bash
pip install "paku[ocr]"  # adds google-cloud-vision
```

Then set credentials — either:
- `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json` (env var), or
- `google_vision.api_key: <key>` in `config.yaml`

Google Cloud Vision free tier covers 1,000 images/month.

## Usage

```bash
# Single image
paku digest screenshot.png

# Directory (processes all images recursively)
paku digest ./screenshots/

# Force a specific extraction mode
paku digest screenshot.png --mode url

# Output formats (repeatable)
paku digest screenshot.png --output json --output txt
```

## Config

Copy `config.yaml.template` to `config.yaml` and fill in your keys. The file is gitignored.

```yaml
google_vision:
  api_key: ""  # or use GOOGLE_APPLICATION_CREDENTIALS env var

anilist:
  base_url: "https://graphql.anilist.co"
  confidence_threshold: 0.8

notion:
  token: ""
  anime_db_id: ""
  url_db_id: ""
  recipe_db_id: ""
```

Everything works with defaults except OCR credentials.

## Tests

```bash
# All tests (262 currently)
python -m pytest

# With coverage
pytest --cov=paku --cov-report=term-missing

# Integration tests (require real OCR credentials + fixture images)
pytest tests/test_google_vision_engine.py -m integration -s
```

Test fixtures go in `tests/fixtures/`. Real screenshots are gitignored — populate them manually.

## Roadmap

| Version | What | Status |
|---------|------|--------|
| v0.1 | Scaffold + OCR baseline | Done |
| v0.2 | URL extractor | Done (gate passed) |
| v0.3 | Anime extractor + AniList | Implemented (gate pending: 57%) |
| v0.4 | Recipe extractor | -- |
| v0.5 | Notion integration | -- |
| v1.0 | Batch processing (3000 screenshots) | -- |

Each version has an explicit gate — a minimum accuracy threshold measured on real screenshots — that must pass before the next version starts.

## Project structure

```
paku/
  cli.py               # Click commands
  pipeline.py           # OCR -> classify -> extract -> output
  config.py             # YAML config loader
  context.py            # Singleton: config + logger + OCR registry
  models.py             # Pydantic v2: OcrResult, ExtractionResult, URLExtractionResult, AnimeExtractionResult
  ocr/
    base.py             # OCREngine ABC
    stub.py             # Fake engine for tests
    google_vision.py    # Google Cloud Vision (document_text_detection)
    router.py           # light/heavy/auto strategy selection
  extractors/
    url.py              # 4-tier URL extraction cascade
    anime.py            # 10-pattern title cascade + AniList enrichment
  outputs/
    json_out.py         # Pretty-printed JSON writer
    txt_out.py          # One-line text writer
```

## License

Personal project. Not currently licensed for external use.
