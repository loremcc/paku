# paku

[![CI](https://github.com/loremcc/paku/actions/workflows/ci.yml/badge.svg)](https://github.com/loremcc/paku/actions/workflows/ci.yml) [![PyPI version](https://img.shields.io/pypi/v/paku.svg)](https://pypi.org/project/paku/) [![Python versions](https://img.shields.io/pypi/pyversions/paku.svg)](https://pypi.org/project/paku/) [![License: MPL-2.0](https://img.shields.io/badge/license-MPL--2.0-brightgreen.svg)](https://github.com/loremcc/paku/blob/main/LICENSE) [![GitHub release](https://img.shields.io/github/v/release/loremcc/paku.svg)](https://github.com/loremcc/paku/releases)

![paku demo](docs/demo.gif)

CLI tool that turns Instagram screenshots into structured data. Feed it a screenshot. It runs OCR (Google Cloud Vision), figures out whether you've shown it an anime recommendation, a GitHub link, or a recipe, pulls the relevant fields, and writes them somewhere you can use.

## What it does

Three extractors:

- **URL** — 4-tier cascade tested on 34 real screenshots. Matches full URLs (github.com, arxiv.org, etc.), spots non-GitHub domains via a curated TLD allowlist, rebuilds GitHub `author/repo` from repo-card layouts, and stubs project-name-only cases for manual review. Survives browser-bar truncation (with or without a visible ellipsis), hyphen-broken URLs, and social-platform false positives. Phase 1 gate: Tier 1 100%, Tier 2-3 71.4%, Tier 4 100%, zero false positives.
- **Anime** — 10-pattern title cascade plus AniList GraphQL enrichment. Strips Instagram UI chrome (15+ filter categories), recognises platform context (AniList app, TikTok, Threads), and pulls every title out of carousel and numbered-list posts. An enhanced Levenshtein ratio (substring containment plus a word-overlap boost) decides auto-accept (>= 0.8) vs review queue. Phase 2 gate: 30/30 = 100% auto-accepted.
- **Recipe** — multilingual ingredient-block detection (English and Italian anchors). Splits every line into quantity, unit, and name. Never stored as "100g" — always `{qty: 100, unit: "g"}`. Handles unicode fractions, wrapped OCR lines, the reversed metric-parens format giallozafferano.com uses, instructions extraction, and source-account detection. Outputs `.txt`, `.csv`, and `.json`. Phase 3 gate: 10/10 = 100%.

Anything the pipeline isn't confident about goes into the review queue instead of getting silently dropped.

`paku serve` starts a local dashboard (FastAPI + vanilla JS SPA) for browsing your collection, uploading screenshots, tracking watch status, and discovering what to watch next. Two recommendation engines: a "For You" panel powered by a local Ollama LLM that analyses your collection context, and a "Similar to…" panel that queries AniList's community recommendation graph. A dedicated Recs tab surfaces both. Import your existing Notion anime database with `paku import-notion` to merge watch statuses and personal scores. SQLite-backed. Runs on 127.0.0.1. No cloud accounts. Phase 5 gate passed.

## Status

**v1.1.0** — three extractors, batch processing, dashboard, Notion status import, local AI-powered semantic recommendations, and dashboard branding are complete. 620 tests pass. CI runs on every push: lint, test matrix (Python 3.11 and 3.12), wheel build. Tagged `v*` pushes auto-publish to PyPI via OIDC Trusted Publishing.

`--smart` flag enables confidence-gated re-run: when fast-path extraction returns confidence < 0.4, the pipeline re-OCRs with a local Ollama VLM (Gemma 4, custom model from `Modelfile.paku`) for richer text and re-extracts. Falls back cleanly if Ollama is unavailable.

Batch mode produces three consolidated outputs: `anime_titles.txt` / `urls.txt` / `recipe_titles.txt` (one entry per line, deduped), plus `anime_export.csv` (9 property columns, ready to import). Per-image JSON is written throughout.

## Install

```bash
pip install paku            # core + stub OCR (for testing)
pip install "paku[ocr]"    # + Google Cloud Vision (real OCR)
pip install "paku[web]"    # + FastAPI dashboard (paku serve)
pip install "paku[smart]"  # + Ollama VLM (--smart flag)
```

Then set OCR credentials — either:
- `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json` (env var), or
- `google_vision.api_key: <key>` in `config.yaml`

Google Cloud Vision free tier covers 1,000 images/month.

### Development install

```bash
git clone https://github.com/loremcc/paku.git
cd paku
pip install -e ".[dev]"
```

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

# Import Notion anime database CSV — merge watch statuses and scores
paku import-notion notion-anime-db.csv --dry-run   # preview matches
paku import-notion notion-anime-db.csv              # commit merge
```

Batch mode writes a `.paku_checkpoint` file in the output directory. Each successfully processed image is recorded there, so `--resume` (the default) skips it on the next run.

Consolidated outputs written after a batch completes:
- `--output txt` → `anime_titles.txt`, `urls.txt`, `recipe_titles.txt` (one entry per line, deduped, sorted)
- `--output csv` with `--mode anime` → `anime_export.csv` (9 property columns, deduped by AniList ID)

## Config

Copy `config.yaml.template` to `config.yaml` and fill in your keys. The file is gitignored.

```yaml
google_vision:
  api_key: ""              # or use GOOGLE_APPLICATION_CREDENTIALS env var
  credentials_file: ""     # path to service account JSON file

anilist:
  base_url: "https://graphql.anilist.co"
  confidence_threshold: 0.8

ollama:
  base_url: "http://localhost:11434"       # or LAN host running Ollama
  ocr_model: "gemma4-paku:latest"          # VLM for smart OCR re-run (built from Modelfile.paku)
  recs_model: "gemma4:26b"                # text LLM for semantic recommendations
```

Everything works with defaults except OCR credentials. The `ollama` section is optional — `--smart` falls back gracefully if Ollama is unavailable. The `recs_model` powers the dashboard Recommendations tab.

## Tests

```bash
# All tests (620 currently)
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
| v0.5 | Batch processing + anime CSV | Done (gate passed 2026-04-24) |
| v0.6 | Dashboard + product identity | Done (gate passed 2026-04-23) |
| v1.0 | Polish + open source | Done (2026-04-26) |
| v1.0.1 | AniList recommendations panel + PyPI auto-publish | Done (2026-04-28) |
| v1.1.0 | Semantic recommendations + personal anime DB + branding | Done (2026-04-30) |

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
    csv_out.py          # Recipe ingredient CSV (per image) + write_anime_csv() (post-batch import)
  inputs/
    notion_import.py    # Notion CSV parser: parse_notion_csv(), Notion URL cleaning, status mapping
  web/
    database.py         # SQLite layer: Database class, user_score column, merge_notion_import
    app.py              # FastAPI factory create_app(db_path), 11 endpoints
    recommendations.py  # Ollama-powered semantic recs: context → prompt → resolve → cache
    static/
      index.html        # Vanilla JS + Tailwind SPA — 5 tabs (Dashboard/Collection/Recs/Add/Review)
Modelfile.paku         # Ollama Modelfile for "gemma4-paku:latest" custom VLM
```

## License

This project is licensed under the [Mozilla Public License 2.0](LICENSE).
