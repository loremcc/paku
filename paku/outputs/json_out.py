from __future__ import annotations

import json
from pathlib import Path


def write_json(
    result_dict: dict,
    screenshot_stem: str,
    output_dir: str | Path,
) -> Path:
    """Write extraction result as pretty-printed JSON.

    Creates output_dir if it doesn't exist.
    Returns the path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{screenshot_stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    return out_path
