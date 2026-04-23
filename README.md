# paku

CLI tool that turns Instagram screenshots into structured data. Built to process a personal backlog of ~3000 screenshots — extracting anime titles, GitHub URLs, and recipes into JSON, CSV, and Notion.

## What it does

You feed it an Instagram screenshot. It runs OCR (Google Cloud Vision), figures out what kind of content it's looking at (anime recommendation, GitHub link, recipe), extracts the relevant data, and writes it out in a usable format.

Three extractors, each purpose-built:

- **URL** (complete, v0.2) — 4-tier extraction cascade validated on 34 real screenshots. Regex-matches full URLs (github.com, arxiv.org, etc.), detects non-GitHub domains via curated TLD allowlist, reconstructs GitHub `author/repo` from repo cards, and stubs project-name-only cases for manual review. Handles browser bar truncation (with and without OCR-visible ellipsis), hyphen-broken URLs, filters social platform URLs, strips noise, and routes uncertain results to the review queue. Phase 1 gate passed: Tier 1 100%, Tier 2-3 71.4%, Tier 4 100%, zero false positives.
- **Anime** (complete, v0.3) — 10-pattern title extraction cascade with AniList GraphQL enrichment. Strips Instagram UI chrome (15+ filter categories), detects platform context (AniList app, TikTok, Threads), handles multi-title posts (carousels, numbered lists). Enhanced Levenshtein ratio (substring containment + word-overlap boost) gates auto-acceptance (>= 0.8) vs review queue. Phase 2 gate passed: 30/30 = 100% auto-accepted.
- **Recipe** (complete, v0.4) — multilingual ingredient block detection (English + Italian anchors), splits each line into quantity + unit + name (never stored as "100g" — always `{qty: 100, unit: "g"}`), handles unicode fractions, wrapped OCR lines, reversed metric-parens format (giallozafferano.com style), music-credit title rejection, instructions extraction, and source account detection. Outputs `.txt` + `.csv` + `.json`. Phase 3 gate passed: 10/10 = 100%.

Anything the pipeline isn't confident about lands in `review_queue.json` instead of being silently discarded.

## Current state

**v0.6 — Dashboard + product identity**

All three extractors are implemented and gate-verified. Batch mode is ready — point `paku digest` at a directory and it processes every image, writes a checkpoint after each one, and picks up where it left off if interrupted. 454 tests pass (2 skipped for missing credentials). Phases 1–3 and 5 gates all passed.

`paku serve` starts a local dashboard (FastAPI + vanilla JS SPA) for browsing the extracted collection, uploading new screenshots, and managing watch status. No cloud accounts required; SQLite-backed, runs on 127.0.0.1.

Batch produces three consolidated outputs: `anime_titles.txt` / `urls.txt` / `recipe_titles.txt` (one entry per line, deduped), plus `anime_export.csv` (9 exact Notion "Full Catalog" property columns, ready to import). Per-image JSON is written throughout.

`--smart` flag enables confidence-gated re-run: when fast-path extraction returns confidence < 0.4, the pipeline re-OCRs with a local Ollama VLM (Gemma 4) for richer text and re-extracts. Falls back cleanly if Ollama is unavailable.

Gate pending: Phase 4 — 1287 images in `input/` processed in one batch run with no crashes, anime CSV importable into Notion Full Catalog.

## Install

```bash
git clone https://github.com/loremcc/paku.git
cd paku
pip install -e ".[dev]"
```

For real OCR (not the stub engine):

```bash
pip install "paku[ocr]"    # adds google-cloud-vision
pip install "paku[smart]"  # enables --smart flag (Ollama VLM re-run)
pip install "paku[web]"    # adds fastapi + uvicorn for paku serve
```

Then set credentials — either:
- `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json` (env var), or
- `google_vision.api_key: <key>` in `config.yaml`

Google Cloud Vision free tier covers 1,000 images/month.

## Usage

```bash
# Single image
paku digest screenshot.png

# Single image — force extraction mode + output formats
paku digest screenshot.png --mode url --output json --output txt

# Smart re-run (re-OCR with Ollama VLM when confidence is low)
paku digest screenshot.png --mode anime --smart

# Batch — directory of images
paku digest ./screenshots/ --mode anime --output csv --output txt --output json

# Batch — resume interrupted run (default behavior: skips already-processed images)
paku digest ./screenshots/ --mode anime --output csv --resume

# Batch — start fresh, ignore checkpoint
paku digest ./screenshots/ --mode anime --output csv --no-resume

# Batch — print breakdown by content type after completion
paku digest ./screenshots/ --report

# Dashboard — browse collection, upload screenshots, manage watch status
paku serve
paku serve --port 8080 --host 127.0.0.1
```

Batch mode writes a `.paku_checkpoint` file in the output directory. Each successfully processed image is recorded there, so `--resume` (the default) skips it on the next run.

Consolidated outputs written after a batch completes:
- `--output txt` → `anime_titles.txt`, `urls.txt`, `recipe_titles.txt` (one entry per line, deduped, sorted)
- `--output csv` with `--mode anime` → `anime_export.csv` (9 Notion property columns, deduped by AniList ID)

## Config

Copy `config.yaml.template` to `config.yaml` and fill in your keys. The file is gitignored.

```yaml
google_vision:
  api_key: ""              # or use GOOGLE_APPLICATION_CREDENTIALS env var
  credentials_file: ""     # or path to service account JSON file

anilist:
  base_url: "https://graphql.anilist.co"
  confidence_threshold: 0.8

ollama:
  base_url: "http://192.168.1.114:11434"  # LAN host running Ollama
  model: "gemma4-paku:latest"             # custom model (see Modelfile.paku)

notion:
  token: ""
  anime_db_id: ""
  url_db_id: ""
  recipe_db_id: ""
```

Everything works with defaults except OCR credentials. The `ollama` section is optional — `--smart` falls back gracefully if Ollama is unavailable.

## Tests

```bash
# All tests (454 currently)
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
| v0.3 | Anime extractor + AniList | Done (gate passed) |
| v0.4 | Recipe extractor | Done (gate passed) |
| v0.5 | Batch processing + anime Notion CSV | Done — gate pending (1287-image input/ run) |
| v0.6 | Dashboard + product identity | Done (gate passed 2026-04-23) |

Each version has an explicit gate — a minimum accuracy threshold or throughput test measured on real screenshots — that must pass before the next version starts.

## Project structure

```
paku/
  cli.py               # Click commands (digest: single + batch, --resume/--no-resume, --report)
  pipeline.py           # OCR -> classify -> extract -> output; process_batch() + BatchReport
  config.py             # YAML config loader
  context.py            # Singleton: config + logger + OCR registry
  models.py             # Pydantic v2: OcrResult, ExtractionResult, URLExtractionResult, AnimeExtractionResult, RecipeExtractionResult, Ingredient
  ocr/
    base.py             # OCREngine ABC
    stub.py             # Fake engine for tests
    google_vision.py    # Google Cloud Vision (document_text_detection)
    ollama.py           # OllamaVLMEngine — smart re-run (stream-parsed NDJSON)
    router.py           # light/heavy/auto/smart strategy selection
  extractors/
    url.py              # 4-tier URL extraction cascade
    anime.py            # 10-pattern title cascade + AniList enrichment
    recipe.py           # multilingual ingredient block detection + qty/unit split
  outputs/
    json_out.py         # Pretty-printed JSON writer (per image)
    txt_out.py          # Per-image text writer + write_batch_txt() (consolidated, deduped)
    csv_out.py          # Recipe ingredient CSV (per image) + write_anime_csv() (post-batch Notion import)
  web/
    database.py         # SQLite layer: Database class, ingest_pipeline_result, Pydantic models
    app.py              # FastAPI factory create_app(db_path), 9 endpoints
    static/
      index.html        # Vanilla JS + Tailwind SPA — Collection, Add, Review, Dashboard tabs
```

## License

This project is licensed under the [Apache License, Version 2.0](LICENSE).
