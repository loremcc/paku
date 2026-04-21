from __future__ import annotations

import os
from pathlib import Path


def write_txt(
    resolved_url: str | None,
    screenshot_stem: str,
    output_dir: str | Path,
) -> Path:
    """Write a single URL (or placeholder) as a one-line text file.

    Creates output_dir if it doesn't exist.
    Returns the path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{screenshot_stem}.txt"
    line = resolved_url if resolved_url else "[no URL resolved — review needed]"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(line + "\n")
    return out_path


def write_batch_txt(entries: list[str], output_path: Path) -> Path:
    """Write batch entries to a single file, one per line, deduped and sorted.

    Uses atomic write: writes to .tmp then os.replace.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / (output_path.name + ".tmp")

    deduped = sorted(set(e for e in entries if e))
    content = "\n".join(deduped) + ("\n" if deduped else "")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, output_path)
    return output_path
