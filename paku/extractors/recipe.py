from __future__ import annotations

# Recipe extractor for Instagram screenshots and web recipe card OCR text.
# Detects ingredient anchor, parses qty/unit splits, assigns confidence, routes review.
# Structural template: follows url.py / anime.py conventions.

import re
from datetime import datetime, timezone
from logging import Logger
from typing import Any

from ..models import Ingredient, RecipeExtractionResult
from ..pipeline import INGREDIENT_ANCHORS
from .url import strip_noise

# --- Unit definitions (longer forms first to ensure correct greedy match) ---

_UNITS = [
    "tablespoons?", "teaspoons?", "tbsp", "tsp",
    r"fl\.?\s*oz",
    "cups?",
    r"cucchiai(?:no|ni|o)?",  # cucchiaio/cucchiai/cucchiaino/cucchiaini (Italian tbsp/tsp)
    "litres?", "liters?", "millilitres?", "milliliters?", "dl", "cl", "ml", "l",
    "kilograms?", "grams?", "milligrams?", "pounds?", "ounces?",
    "kg", "mg", "oz", "lbs?", "g",
    "large", "medium", "small", "big",
    r"pizzic(?:o|hi)",  # pizzico/pizzichi (Italian: pinch)
    r"pinch(?:es)?", r"handful(?:s)?", r"bunch(?:es)?",
    r"cloves?", r"sprigs?", r"slices?", r"cans?", r"pieces?",
    r"bags?", r"knobs?", r"strips?", r"stalks?", r"heads?",
    r"rashers?", r"fillets?",
]

_INGREDIENT_ANCHOR_RE = re.compile("|".join(INGREDIENT_ANCHORS), re.IGNORECASE)

_INSTRUCTION_ANCHOR_RE = re.compile(
    r"(?:method|instructions?|directions?|preparation"
    r"|how\s+to\s+(?:make|prepare)|steps?|istruzioni|procedimento"
    r"|preparazione|la\s+ricetta\s+passo|passo\s+passo|come\s+si\s+prepara)\s*:?",
    re.IGNORECASE,
)

_UNIT_START_RE = re.compile(r"^(" + "|".join(_UNITS) + r")\b", re.IGNORECASE)

# Unicode fractions → float
_FRACTIONS: dict[str, float] = {
    "½": 0.5, "¼": 0.25, "¾": 0.75,
    "⅓": 1 / 3, "⅔": 2 / 3,
    "⅛": 0.125, "⅜": 0.375,
}
_FRACTION_CHARS = "".join(_FRACTIONS)

# Quantity: most-specific first to prevent greedy integer from eating vulgar/mixed fractions
_QUANTITY_RE = re.compile(
    rf"^(\d+/\d+"
    rf"|\d+\s*[{re.escape(_FRACTION_CHARS)}]"
    rf"|[{re.escape(_FRACTION_CHARS)}]"
    r"|\d+(?:[.,]\d+)?"
    r"|\d+)\s*"
)

# Number (possibly with decimal) directly followed by optional whitespace then unit
_ATTACHED_UNIT_RE = re.compile(
    r"^(\d+(?:[.,]\d+)?)\s*(" + "|".join(_UNITS) + r")\b\s*(.*)",
    re.IGNORECASE,
)

# "a/an <unit> [of] <name>"
_WORD_QTY_RE = re.compile(
    r"^(?:a|an)\s+(" + "|".join(_UNITS) + r")\s+(?:of\s+)?(.+)",
    re.IGNORECASE,
)

# Trailing qty-unit: "Ingredient name 500 g" or "Ingredient name 2 tbsp"
# Used as fallback when no leading quantity is found (giallozafferano format)
_TRAILING_QTY_RE = re.compile(
    r"^(.*?)\s+(\d+(?:[.,]\d+)?)\s*(" + "|".join(_UNITS) + r")\s*$",
    re.IGNORECASE,
)

_LIST_PREFIX_RE = re.compile(r"^[\s•\-\*·–—+]+|^\d+[.)]\s+")

# Parenthesized metric equivalent: "(450 g)", "(150 ml)" — giallozafferano / recipe website reversed format
_METRIC_PARENS_RE = re.compile(r"\(\s*(\d+(?:[.,]\d+)?)\s*(g|kg|ml|l)\s*\)", re.IGNORECASE)

# Strip trailing garbled imperial qty+unit from reversed-format lines after metric parens are removed
# e.g. "Whole milk 134 cup" → "Whole milk", "Egg yolks 2 12 tbsp" → "Egg yolks"
_IMPERIAL_TRAIL_RE = re.compile(
    r"\s+\d[\d\s/.,¾½¼⅔⅓]*\s*(?:cups?|tablespoons?|teaspoons?|tbsp|tsp|fl\.?\s*oz)\s*$",
    re.IGNORECASE,
)

_NOTES_PARENS_RE = re.compile(r"\s*\(([^)]+)\)\s*$")
_NOTES_COMMA_RE = re.compile(
    r",\s*(to\s+taste|optional|for\s+garnish|to\s+serve)\s*$", re.IGNORECASE
)

_CHROME_LINE_RE = re.compile(
    r"^(?:[@#]|Follow|Send message|Add comment|Liked by|See translation|Suggested"
    r"|Posts|Tagged|Reels|IGTV"
    r"|segui\s+@|salva\s+la\s+ricetta"
    r"|aggiungi\s+alla\s+lista|Not interested|Stampa|stampa\s+la\s+ricetta"
    r"|vieni\s+a\s+trovarci|trova\s+il\s+negozio"
    r"|↗)",
    re.IGNORECASE,
)

_HANDLE_RE = re.compile(r"^@?([a-zA-Z0-9_.][a-zA-Z0-9_.]{2,29})$")
# "username Follow" / "username Following" — one OCR line containing handle + chrome button
_HANDLE_FOLLOW_RE = re.compile(
    r"^@?([a-zA-Z0-9_.][a-zA-Z0-9_.]{2,29})\s+Follow(?:ing)?\s*$", re.IGNORECASE
)
_CHROME_HANDLES = frozenset({
    "follow", "following", "lowing", "home", "inbox", "explore", "profile", "message",
    "share", "comments", "likes", "reels", "friends",
    "posts", "stampa", "safari", "chrome", "firefox", "edge", "search", "print",
    "instagram",
})

# Common TLDs that mark a string as a domain, not an Instagram handle
_COMMON_TLDS = frozenset({
    "com", "org", "net", "io", "it", "fr", "de", "uk", "gov", "edu",
    "co", "app", "dev", "ru", "jp", "cn", "br", "es", "nl", "pl",
})

# Matches a handle embedded in a line decorated with non-ASCII symbols (e.g. "☐ username ✪")
_HANDLE_DECORATED_RE = re.compile(r"^[^\x20-\x7E]*\s*@?([a-zA-Z0-9_.]{3,30})\s*[^\x20-\x7E]*$")

_UNIT_CANONICAL: dict[str, str] = {
    "tablespoon": "tbsp", "tablespoons": "tbsp",
    "teaspoon": "tsp", "teaspoons": "tsp",
    "gram": "g", "grams": "g",
    "kilogram": "kg", "kilograms": "kg",
    "milligram": "mg", "milligrams": "mg",
    "millilitre": "ml", "milliliter": "ml", "millilitres": "ml", "milliliters": "ml",
    "litre": "l", "liter": "l", "litres": "l", "liters": "l",
    "ounce": "oz", "ounces": "oz",
    "cucchiaio": "tbsp", "cucchiai": "tbsp",
    "cucchiaino": "tsp", "cucchiaini": "tsp",
    "pizzico": "pinch", "pizzichi": "pinch",
    "pound": "lb", "pounds": "lb", "lbs": "lb",
    "cup": "cup", "cups": "cup",
    "clove": "clove", "cloves": "clove",
    "sprig": "sprig", "sprigs": "sprig",
    "slice": "slice", "slices": "slice",
    "can": "can", "cans": "can",
    "piece": "piece", "pieces": "piece",
    "bag": "bag", "bags": "bag",
    "handful": "handful", "handfuls": "handful",
    "bunch": "bunch", "bunches": "bunch",
    "pinch": "pinch", "pinches": "pinch",
    "knob": "knob", "knobs": "knob",
    "strip": "strip", "strips": "strip",
    "stalk": "stalk", "stalks": "stalk",
    "head": "head", "heads": "head",
    "rasher": "rasher", "rashers": "rasher",
    "fillet": "fillet", "fillets": "fillet",
}


# --- Helpers ---


def _is_domain(handle: str) -> bool:
    """Return True if handle looks like a website domain (ends with a common TLD)."""
    if "." not in handle:
        return False
    parts = handle.rsplit(".", 1)
    return parts[-1].lower() in _COMMON_TLDS


def _handle_from_line(stripped: str) -> str | None:
    """Extract Instagram handle from a line in any of three formats:
    - bare handle: "username"
    - handle + button: "username Follow"
    - decorated: "☐ username ✪"
    Returns None if line is not a handle line.
    """
    # Reject lines that are ingredient anchors
    if _INGREDIENT_ANCHOR_RE.search(stripped):
        return None
    m = _HANDLE_FOLLOW_RE.match(stripped)
    if m:
        h = m.group(1)
        if h.lower() not in _CHROME_HANDLES and not _is_domain(h) and not (h == h.upper() and h.isalpha()):
            return h
    m = _HANDLE_RE.match(stripped)
    if m:
        h = m.group(1)
        if h.lower() not in _CHROME_HANDLES and not _is_domain(h) and not (h == h.upper() and h.isalpha()):
            return h
    m = _HANDLE_DECORATED_RE.match(stripped)
    if m:
        h = m.group(1)
        if h.lower() not in _CHROME_HANDLES and not _is_domain(h) and not (h == h.upper() and h.isalpha()):
            return h
    return None


def _parse_qty(s: str) -> float | None:
    """Parse quantity string to float. Handles decimals, fractions, unicode fractions."""
    s = s.strip()
    # Mixed number: "2½", "2 ½"
    m = re.match(rf"^(\d+)\s*([{re.escape(_FRACTION_CHARS)}])$", s)
    if m:
        return float(m.group(1)) + _FRACTIONS[m.group(2)]
    # Standalone unicode fraction
    if s in _FRACTIONS:
        return _FRACTIONS[s]
    # Vulgar fraction
    if "/" in s:
        parts = s.split("/", 1)
        try:
            return float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError):
            return None
    # Decimal (handle European comma separator)
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def _normalize_unit(unit_str: str) -> str:
    """Return canonical unit abbreviation."""
    lower = unit_str.lower().strip()
    return _UNIT_CANONICAL.get(lower, lower)


def _clean_non_ascii(text: str) -> str:
    """Replace non-ASCII chars (except known fractions) with space to prevent qty/unit merging."""
    result: list[str] = []
    for ch in text:
        if ord(ch) < 128 or ch in _FRACTIONS:
            result.append(ch)
        else:
            result.append(" ")
    return re.sub(r" {2,}", " ", "".join(result)).strip()


def _extract_notes(name: str) -> tuple[str, str | None]:
    """Strip trailing parenthesized or comma-delimited notes from name."""
    m = _NOTES_PARENS_RE.search(name)
    if m:
        return name[:m.start()].strip(), m.group(1).strip()
    m = _NOTES_COMMA_RE.search(name)
    if m:
        return name[:m.start()].strip(), m.group(1).strip()
    return name.strip(), None


def _parse_ingredient_line(line: str) -> Ingredient | None:
    """Parse one ingredient line into name/qty/unit/notes. Returns None on failure."""
    cleaned = _LIST_PREFIX_RE.sub("", line, count=1).strip()
    if len(cleaned) < 2:
        return None

    # Strip OCR bullet artifact: lowercase 'o' before uppercase (giallozafferano emoji → 'o')
    if re.match(r"^o\s+(?=[A-Z])", cleaned):
        cleaned = cleaned[2:].strip()

    cleaned = _clean_non_ascii(cleaned)
    if not cleaned:
        return None

    # Reject section headers and chrome UI lines
    if _INGREDIENT_ANCHOR_RE.search(cleaned) or _INSTRUCTION_ANCHOR_RE.search(cleaned):
        return None
    if _CHROME_LINE_RE.match(cleaned):
        return None

    # Reject lines ending with ":" (section sub-headers like "For the sauce:")
    if cleaned.endswith(":"):
        return None

    # Reject all-caps multi-word lines (section headers like "FOR THE CARAMEL")
    words = cleaned.split()
    if len(words) >= 2 and cleaned == cleaned.upper() and cleaned.replace(" ", "").isalpha():
        return None

    # Reject calorie/macro lines
    if re.search(r"\bkcal\b|\bcalories\b|\bcalorie\b", cleaned, re.IGNORECASE):
        return None

    qty: float | None = None
    unit: str | None = None
    name: str = cleaned

    m = _WORD_QTY_RE.match(cleaned)
    if m:
        unit = _normalize_unit(m.group(1))
        name = m.group(2).strip()
    else:
        m = _ATTACHED_UNIT_RE.match(cleaned)
        if m:
            qty = _parse_qty(m.group(1))
            unit = _normalize_unit(m.group(2))
            name = m.group(3).strip()
        else:
            m_qty = _QUANTITY_RE.match(cleaned)
            if m_qty:
                qty = _parse_qty(m_qty.group(1))
                remainder = cleaned[m_qty.end():].strip()
                m_unit = _UNIT_START_RE.match(remainder)
                if m_unit:
                    unit = _normalize_unit(m_unit.group(1))
                    name = remainder[m_unit.end():].strip()
                else:
                    name = remainder
            else:
                # Metric-in-parens reversed format: "Name garbled_qty imperial_unit (metric_qty g)"
                # e.g. "Whole milk 134 cup (450 g)" — OCR garbles ¾→34, prefer reliable metric value
                m_metric = _METRIC_PARENS_RE.search(cleaned)
                if m_metric:
                    metric_qty = _parse_qty(m_metric.group(1))
                    metric_unit = _normalize_unit(m_metric.group(2))
                    if metric_qty is not None:
                        # Strip all parenthesized groups (metric + secondary notes like "- (about 4 medium)")
                        without_parens = re.sub(r"\s*[-–—]?\s*\([^)]+\)", "", cleaned).strip()
                        # Strip trailing garbled imperial qty+unit (e.g. "134 cup", "2 12 tbsp")
                        clean_name = _IMPERIAL_TRAIL_RE.sub("", without_parens).strip()
                        # Strip residual trailing lone digit (mixed-number OCR fragment: "Whole milk 1")
                        clean_name = re.sub(r"\s+\d+\s*$", "", clean_name).strip() or without_parens
                        name, notes = _extract_notes(clean_name)
                        if name:
                            return Ingredient(name=name, quantity=metric_qty, unit=metric_unit, notes=notes)

                # Trailing-qty fallback: "Ingredient name 500 g" or "Ingredient name ¾ cup"
                cleaned_for_trail = re.sub(r"\s*[-–—]?\s*\([^)]+\)\s*$", "", cleaned)
                m_trail = _TRAILING_QTY_RE.match(cleaned_for_trail)
                if m_trail:
                    name = m_trail.group(1).strip()
                    qty = _parse_qty(m_trail.group(2))
                    unit = _normalize_unit(m_trail.group(3))
                else:
                    # Bare count: "Vanilla bean 1" → name="Vanilla bean", qty=1, unit=None
                    m_bare = re.match(r"^([A-Za-z][A-Za-z\s]+?)\s+(\d+)\s*$", cleaned)
                    if m_bare:
                        bare_name = m_bare.group(1).strip()
                        bare_words = bare_name.split()
                        if len(bare_words) >= 2 and all(w.isalpha() for w in bare_words):
                            name = bare_name
                            qty = float(m_bare.group(2))

    name, notes = _extract_notes(name)
    if not name:
        return None
    return Ingredient(name=name, quantity=qty, unit=unit, notes=notes)


def _find_block_end(lines: list[str], start: int) -> int:
    """Return index past the last ingredient line starting from `start`."""
    end = start
    empty_run = 0
    for i in range(start, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            empty_run += 1
            if empty_run >= 2:
                break
            end = i + 1
            continue
        empty_run = 0
        if _INSTRUCTION_ANCHOR_RE.search(stripped):
            break
        if _CHROME_LINE_RE.match(stripped):
            break
        # "You might also like" — signals sidebar/related-posts section
        if re.search(r"you\s+might\s+also\s+like", stripped, re.IGNORECASE):
            break
        # Numbered instruction step: "1. Peel the..." (signals recipe body, not ingredients)
        if re.match(r"^\d+\.\s+[A-Z]", stripped):
            break
        # Website domain line (e.g. "www.giallozafferano.it") — end of recipe card content
        if re.match(r"^(?:www\.)?[\w-]+\.\w{2,}\.\w{2,3}$", stripped, re.IGNORECASE):
            break
        # Italian/web end-of-content signals
        if re.search(
            r"aggiungi\s+alla\s+lista|trova\s+il\s+negozio|vieni\s+a\s+trovarci",
            stripped, re.IGNORECASE,
        ):
            break
        # Dots-only separator artifact (". . .") — signals end of content block
        if re.match(r"^[.\s]+$", stripped) and len(stripped) >= 3:
            break
        # Long prose line without list/digit prefix → instruction paragraph
        if len(stripped) > 60 and not re.match(r"^[•\-\*·–—\d+]", stripped):
            break
        end = i + 1
    return end


def _find_first_ingredient_line(lines: list[str]) -> int:
    """Return the index of the first line that looks like an ingredient.

    Used when no ingredient anchor is found. Skips chrome, titles, and handle lines.
    Falls back to 0 if no clear ingredient start is found.
    """
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if _CHROME_LINE_RE.match(stripped):
            continue
        if _INGREDIENT_ANCHOR_RE.search(stripped):
            continue
        if _INSTRUCTION_ANCHOR_RE.search(stripped):
            continue
        if _handle_from_line(stripped) is not None:
            continue
        # Skip timestamps (e.g., "13:21" from Instagram status bar)
        if re.match(r"^\d{1,2}:\d{2}", stripped):
            continue
        # Skip engagement metric lines (e.g., "490 13 3 143" — likes/comments/shares/saves)
        if re.match(r"^\d+(?:\s+\d+){2,}\s*$", stripped):
            continue
        # Line starts with a digit → check it parses as a real ingredient (>= 3 alpha chars in name)
        if re.match(r"^\d", stripped):
            parsed = _parse_ingredient_line(stripped)
            alpha_count = sum(1 for c in (parsed.name if parsed else "") if c.isalpha())
            if parsed is not None and alpha_count >= 3:
                return i
            continue
        # Line starts with a list marker followed by content (allow no space: "-1 lb")
        if re.match(r"^[•\-\*·–—]\s*\w", stripped):
            return i
        # OCR bullet artifact ("o CapitalWord")
        if re.match(r"^o\s+[A-Z]", stripped):
            return i
        # Word quantity ("a clove of")
        if _WORD_QTY_RE.match(stripped):
            return i
    return 0


def _find_ingredient_block(lines: list[str]) -> tuple[int | None, int, int]:
    """Locate ingredient anchor and block boundaries.

    Returns (anchor_idx | None, ing_start, ing_end).
    When no anchor is found, uses heuristic to find first ingredient line.
    """
    anchor_idx: int | None = None
    for i, line in enumerate(lines):
        if _INGREDIENT_ANCHOR_RE.search(line):
            anchor_idx = i
            break

    if anchor_idx is not None:
        start = anchor_idx + 1
        end = _find_block_end(lines, start)
    else:
        start = _find_first_ingredient_line(lines)
        end = _find_block_end(lines, start)

    return anchor_idx, start, end


def _extract_title(
    lines: list[str],
    anchor_idx: int | None,
    source_account: str | None = None,
    ing_start: int | None = None,
) -> str | None:
    """Return the most prominent content line before the ingredient block."""
    if anchor_idx is not None:
        search_end = anchor_idx
    elif ing_start is not None:
        search_end = ing_start
    else:
        search_end = min(15, len(lines))
    candidates: list[str] = []
    for line in lines[:search_end]:
        stripped = line.strip()
        # Strip handle prefix from combined lines like "ritas.recipes RECIPE TITLE"
        if source_account:
            stripped = re.sub(
                r"^@?" + re.escape(source_account) + r"\s+",
                "",
                stripped,
                flags=re.IGNORECASE,
            )
        if len(stripped) < 4:
            continue
        if _CHROME_LINE_RE.match(stripped):
            continue
        # Skip pure number/punctuation and timestamp lines
        if re.match(r"^[\d.,\s]+$", stripped):
            continue
        if re.match(r"^\d{1,2}:\d{2}(?:\s*[APap][Mm])?$", stripped):
            continue
        # Skip calorie/macro lines
        if re.search(r"\bkcal\b|\bcalories\b|\bcalorie\b", stripped, re.IGNORECASE):
            continue
        # "You might also like" — stop processing; everything after is navigation
        if re.search(r"you might also like", stripped, re.IGNORECASE):
            break
        # Skip handle lines in all formats (bare, handle+Follow, ☐ handle ✪)
        if _handle_from_line(stripped) is not None:
            continue
        # Skip Instagram music credit lines: "Artist · Song" (middle-dot separator)
        if "·" in stripped:
            continue
        # Skip truncated caption fragments ending with ")" but no opening "(" — e.g. "below)"
        if stripped.endswith(")") and "(" not in stripped:
            continue
        candidates.append(stripped)
    if not candidates:
        return None

    def _title_score(s: str) -> int:
        words = s.split()
        # Prefer ALL-CAPS recipe title words (common in Instagram post format)
        upper_bonus = 3 * sum(1 for w in words if w.isupper() and w.isalpha() and len(w) > 2)
        return len(s) + upper_bonus

    return max(candidates, key=_title_score)


def _extract_source_account(lines: list[str], search_end: int | None = None) -> str | None:
    """Extract Instagram account handle from OCR lines.

    Two-pass scan:
    - Pass 1: "username Follow/Following" anywhere in scan range (high-confidence chrome signal)
    - Pass 2: bare/decorated handle in first 8 lines only (avoids website nav fragments)

    search_end: only scan lines[:search_end]. When None, scans all lines.
    """
    scan_lines = lines[:search_end] if search_end is not None else lines

    for line in scan_lines:
        stripped = line.strip()
        m = _HANDLE_FOLLOW_RE.match(stripped)
        if m:
            h = m.group(1)
            if h.lower() not in _CHROME_HANDLES and not _is_domain(h) and not (h == h.upper() and h.isalpha()):
                return h

    for line in scan_lines[:8]:
        stripped = line.strip()
        if _HANDLE_FOLLOW_RE.match(stripped):
            continue
        if _INGREDIENT_ANCHOR_RE.search(stripped):
            continue
        m = _HANDLE_RE.match(stripped)
        if m:
            h = m.group(1)
            if h.lower() not in _CHROME_HANDLES and not _is_domain(h) and not (h == h.upper() and h.isalpha()):
                return h
        m = _HANDLE_DECORATED_RE.match(stripped)
        if m:
            h = m.group(1)
            if h.lower() not in _CHROME_HANDLES and not _is_domain(h) and not (h == h.upper() and h.isalpha()):
                return h

    return None


def _extract_instructions(lines: list[str], ing_end: int) -> str | None:
    """Extract instruction text following the ingredient block."""
    if ing_end >= len(lines):
        return None
    inst_start = ing_end
    for i in range(ing_end, len(lines)):
        if _INSTRUCTION_ANCHOR_RE.search(lines[i].strip()):
            inst_start = i + 1
            break
    inst_lines = [l.strip() for l in lines[inst_start:] if l.strip()]
    return "\n".join(inst_lines) if inst_lines else None


def _join_wrapped_lines(lines: list[str]) -> list[str]:
    """Join OCR lines where the previous line has an unclosed parenthesis.

    Handles OCR line-breaks mid-ingredient like:
      "30 ml di sciroppo d'agave (o dolcificante a"
      "piacere)"
    → "30 ml di sciroppo d'agave (o dolcificante a piacere)"
    """
    result: list[str] = []
    for line in lines:
        if result and result[-1].count("(") > result[-1].count(")"):
            result[-1] = result[-1] + " " + line.strip()
        else:
            result.append(line)
    return result


def _assign_confidence(
    anchor_idx: int | None,
    ingredients: list[Ingredient],
) -> tuple[float, bool]:
    """Return (confidence, needs_review) based on anchor presence and parse quality."""
    if anchor_idx is not None:
        if len(ingredients) >= 2 and all(ing.name for ing in ingredients):
            return 0.9, False
        return 0.7, True
    if ingredients:
        return 0.5, True
    return 0.1, True


# --- Main entry point ---


def extract(
    ocr_text: str,
    screenshot_path: str,
    config: dict[str, Any],
    logger: Logger,
) -> RecipeExtractionResult:
    """Extract structured recipe data from OCR text.

    Always returns a RecipeExtractionResult. Low-confidence results have needs_review=True.
    """
    now = datetime.now(timezone.utc).isoformat()

    cleaned = strip_noise(ocr_text)
    logger.debug(f"[recipe] noise stripped: {len(ocr_text)} -> {len(cleaned)} chars")

    lines = _join_wrapped_lines(cleaned.splitlines())

    anchor_idx, ing_start, ing_end = _find_ingredient_block(lines)
    logger.debug(f"[recipe] anchor_idx={anchor_idx}, block={ing_start}:{ing_end}")

    ingredients: list[Ingredient] = []
    for line in lines[ing_start:ing_end]:
        ing = _parse_ingredient_line(line)
        if ing is not None:
            ingredients.append(ing)

    logger.debug(f"[recipe] parsed {len(ingredients)} ingredients")

    # Compute source_account first so title extraction can strip handle prefix
    source_account = _extract_source_account(
        lines,
        search_end=anchor_idx if anchor_idx is not None else min(15, len(lines)),
    )
    title = _extract_title(lines, anchor_idx, source_account=source_account, ing_start=ing_start)
    instructions = _extract_instructions(lines, ing_end)

    confidence, needs_review = _assign_confidence(anchor_idx, ingredients)
    dedup_key = title.lower().strip() if title else None

    return RecipeExtractionResult(
        confidence=confidence,
        needs_review=needs_review,
        source_screenshot=screenshot_path,
        extracted_at=now,
        title=title,
        source_account=source_account,
        ingredients=ingredients,
        instructions=instructions,
        dedup_key=dedup_key,
    )
