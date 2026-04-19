from __future__ import annotations

# Writes recipe ingredient rows to CSV for Notion bulk import.
# Atomic write (tmp → os.replace) matches review_queue.json pattern.
# Column headers match Notion property names exactly (case-sensitive).

import csv
import io
import os
from pathlib import Path


def write_csv(ingredients: list[dict], screenshot_stem: str, output_dir: str) -> Path:
    """Write ingredient rows to <output_dir>/<screenshot_stem>.csv.

    Columns: ingredient, quantity, unit, notes
    Uses atomic write: writes to .tmp then os.replace to avoid partial files.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{screenshot_stem}.csv"
    tmp_path = out_dir / f"{screenshot_stem}.csv.tmp"

    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["ingredient", "quantity", "unit", "notes"],
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in ingredients:
        qty = row.get("quantity")
        writer.writerow({
            "ingredient": row.get("name", ""),
            "quantity": str(qty) if qty is not None else "",
            "unit": row.get("unit") or "",
            "notes": row.get("notes") or "",
        })

    tmp_path.write_text(buf.getvalue(), encoding="utf-8")
    os.replace(tmp_path, out_path)
    return out_path
