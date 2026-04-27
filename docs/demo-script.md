# paku demo script

**Terminal:** 100 cols × 30 rows  
**Fixture:** `tests/fixtures/anime/IMG_5417.PNG`  
**Toolchain:** programmatic `.cast` generation (asciinema v2 NDJSON) → `tools/agg.exe` → GIF

## Chosen fixture

`tests/fixtures/anime/IMG_5417.PNG` — Instagram post from `@animecultivated` featuring
"RAINBOW: NISHA ROKUBŌ NO SHICHININ". Produces confidence 0.9, AniList match
(canonical: "Rainbow", romaji: "RAINBOW: Nisha Rokubou no Shichinin", score 8.2, MADHOUSE).

## Demo sequence (2 commands)

```
$ paku digest tests/fixtures/anime/IMG_5417.PNG --mode anime --output json
--- IMG_5417.PNG ---
  screen_type : post
  content_type: anime
  engine      : google_vision
  confidence  : 0.9
  ocr_text    :
    ANIMECULTIVATED
    RAINBOW: NISHA ROKUBŌ NO SHICHININ
    Completed  ·  26 EPS  ·  MAL 8.46|2010
    ...

$ cat output/IMG_5417.json
{
  "extractor": "anime",
  "confidence": 0.9,
  "canonical_title": "Rainbow",
  "romaji": "RAINBOW: Nisha Rokubou no Shichinin",
  "genres": ["Drama", "Psychological", "Thriller"],
  "score": 8.2,
  "studios": ["MADHOUSE"],
  "debut_year": 2010,
  "anilist_url": "https://anilist.co/anime/6114"
}
```

## Generation

Download `agg` (not committed — too large for git):

```bash
# Windows x86_64
curl -L -o tools/agg.exe https://github.com/asciinema/agg/releases/download/v1.7.0/agg-x86_64-pc-windows-msvc.exe
mkdir -p tools
```

Then generate:

```bash
python docs/gen_cast.py        # writes docs/demo.cast
tools/agg.exe docs/demo.cast docs/demo.gif --font-size 14 --speed 1.5
```

## Notes

- `asciinema rec` is Linux-only (`fcntl` missing on Windows) — `.cast` generated from
  captured `paku` output with scripted per-line timing.
- `docs/demo.cast` is gitignored (raw recording, only the GIF is committed).
- Target GIF size: < 3 MB. Re-run agg with `--speed 2.0` if > 5 MB.
