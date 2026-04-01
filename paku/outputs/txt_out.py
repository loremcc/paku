from __future__ import annotations

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
