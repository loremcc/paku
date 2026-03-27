# tests/fixtures/

Real Instagram screenshots for Phase 0 OCR evaluation.

**This directory must be populated manually.** Do not commit screenshots here — add this
directory to `.gitignore` if the screenshots contain personal content.

## Required fixture images (minimum 5)

| Filename | Description | Expected OCR challenge |
|---|---|---|
| `test_anime_story.png` | Dark story with overlay text (anime title) | Low contrast, overlay text on video frame |
| `test_anime_post.png` | Post caption in Italian (anime mention) | UI chrome mixed with caption text |
| `test_url_feed_card.png` | Feed card with GitHub / HuggingFace URL | URL in embedded card, smaller font |
| `test_recipe_caption.png` | Post caption with ingredient list | Multi-line text, possible emoji/bullets |
| `test_hard_dark.png` | Ambiguous / hard case (dark background, small text) | Worst-case OCR scenario |

## Phase 0 gate

Run integration tests with:
```
pytest tests/test_google_vision_engine.py -m integration -s
```

The `-s` flag prints the per-fixture Phase 0 progress report to stdout.

**Gate criterion:** 15 out of 20 real screenshots must produce ≥ 15 meaningful characters
via Google Cloud Vision. Individual fixture tests log PASS/FAIL but do not fail the suite —
evaluate the aggregate count manually.

## Credentials

Set one of:
- `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json` (env var)
- `google_vision.api_key: <key>` in `config.yaml` (API key fallback)

Free tier: 1,000 image annotations/month — sufficient for Phase 0 (20 screenshots).
