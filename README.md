# paku

CLI tool that turns Instagram screenshots into structured data. Built to process a personal backlog of ~3000 screenshots — extracting anime titles, GitHub URLs, and recipes into JSON, CSV, and Notion.

## What it does

You feed it an Instagram screenshot. It runs OCR (Google Cloud Vision), figures out what kind of content it's looking at (anime recommendation, GitHub link, recipe), extracts the relevant data, and writes it out in a usable format.

Three extractors, each purpose-built:

- **URL** — regex-matches `github.com`, `arxiv.org`, `huggingface.co` and other domains directly from OCR text. Falls back to keyword-based web search when no URL is visible. Flags search-resolved URLs for manual review.
- **Anime** — pulls anime/manga titles from Italian and English captions, queries AniList for canonical metadata (title, episodes, genres, score, cover art). Levenshtein ratio gates auto-acceptance vs review queue.
- **Recipe** — detects ingredient blocks, splits each line into quantity + unit + name (never stored as "100g" — always `{qty: 100, unit: "g"}`).

Anything the pipeline isn't confident about lands in `review_queue.json` instead of being silently discarded.

## Current state

**v0.1 — scaffold + OCR baseline (complete)**

The pipeline runs end-to-end: load image → preprocess → OCR → classify screen type + content type → return structured result. Google Cloud Vision produces real OCR text from all tested Instagram screenshot types. 84 tests pass.

No extractors are implemented yet. `paku digest` currently returns `status: "pending_extraction"` for every image. URL extractor is next.

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
# All tests (84 currently)
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
| v0.2 | URL extractor | Next |
| v0.3 | Anime extractor + AniList | — |
| v0.4 | Recipe extractor | — |
| v0.5 | Notion integration | — |
| v1.0 | Batch processing (3000 screenshots) | — |

Each version has an explicit gate — a minimum accuracy threshold measured on real screenshots — that must pass before the next version starts.

## Project structure

```
paku/
  cli.py               # Click commands
  pipeline.py           # OCR → classify → extract → output
  config.py             # YAML config loader
  context.py            # Singleton: config + logger + OCR registry
  models.py             # Pydantic v2: OcrResult, OcrBlock, BoundingBox
  ocr/
    base.py             # OCREngine ABC
    stub.py             # Fake engine for tests
    google_vision.py    # Google Cloud Vision (document_text_detection)
    router.py           # light/heavy/auto strategy selection
  extractors/           # URL, anime, recipe (v0.2+)
  outputs/              # JSON, TXT, CSV writers (v0.2+)
```

## License

Personal project. Not currently licensed for external use.
