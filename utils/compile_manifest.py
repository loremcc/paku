#!/usr/bin/env python3
"""
Compile test fixture manifest.

Scans tests/fixtures/{group}/ subdirectories for image files, optionally
renames them to a consistent convention, and generates/updates manifest.json
per group.

Usage:
    python utils/compile_manifest.py                    # Scan all groups
    python utils/compile_manifest.py --rename           # Rename + generate
    python utils/compile_manifest.py --dry-run          # Preview without applying
    python utils/compile_manifest.py --group anime      # Process only 'anime'
    python utils/compile_manifest.py --init anime       # Create new group

Structure:
    tests/fixtures/
    ├── anime/
    │   ├── manifest.json       # Auto-generated
    │   ├── *.png / *.jpg
    │   └── expected/
    ├── url/
    └── recipe/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"

RESERVED_DIRS = {"output", "expected", "__pycache__"}


def get_groups(fixtures_dir: Path) -> list[str]:
    groups = []
    for d in fixtures_dir.iterdir():
        if d.is_dir() and d.name not in RESERVED_DIRS and not d.name.startswith("."):
            groups.append(d.name)
    return sorted(groups)


def get_image_files(group_dir: Path) -> list[Path]:
    files = [f for f in group_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=lambda p: p.name.lower())


def compute_hash(file_path: Path, algorithm: str = "md5") -> str:
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def parse_existing_name(filename: str) -> dict[str, Any]:
    stem = Path(filename).stem.lower()
    match = re.match(r"^([a-z]+)_([a-z]+)_([a-z]+)_(\d+)$", stem)
    if match:
        return {
            "source": match.group(1),
            "type": match.group(2),
            "lang": match.group(3),
            "seq": int(match.group(4)),
        }
    return {}


def generate_new_name(index: int, group: str, prefix: str = "ig", lang: str = "ja", ext: str = ".png") -> str:
    return f"{prefix}_{group}_{lang}_{index:04d}{ext.lower()}"


def rename_files(
    group_dir: Path,
    group: str,
    prefix: str = "ig",
    lang: str = "ja",
    dry_run: bool = False,
) -> list[dict[str, str]]:
    files = sorted(get_image_files(group_dir), key=lambda p: p.stat().st_mtime)
    renames = []
    for idx, file_path in enumerate(files, start=1):
        new_name = generate_new_name(index=idx, group=group, prefix=prefix, lang=lang, ext=file_path.suffix)
        new_path = file_path.parent / new_name
        if file_path.name == new_name:
            continue
        if new_path.exists() and new_path != file_path:
            file_hash = compute_hash(file_path)
            new_name = generate_new_name(
                index=idx, group=f"{group}_{file_hash}", prefix=prefix, lang=lang, ext=file_path.suffix
            )
            new_path = file_path.parent / new_name
        renames.append({"old": file_path.name, "new": new_name})
        if not dry_run:
            file_path.rename(new_path)
    return renames


def build_manifest(group_dir: Path, group: str, existing_manifest: dict | None = None) -> dict[str, Any]:
    files = get_image_files(group_dir)
    existing_samples: dict[str, dict] = {}
    if existing_manifest and "samples" in existing_manifest:
        for s in existing_manifest["samples"]:
            existing_samples[s["file"]] = s

    samples = []
    for file_path in files:
        filename = file_path.name
        sample: dict[str, Any] = {
            "file": filename,
            "hash": compute_hash(file_path),
        }
        parsed = parse_existing_name(filename)
        sample["source"] = parsed.get("source", "instagram")
        sample["lang"] = parsed.get("lang", "ja")
        if filename in existing_samples:
            for key in ("tags", "difficulty", "sets", "notes", "lang", "source"):
                if key in existing_samples[filename]:
                    sample[key] = existing_samples[filename][key]
        samples.append(sample)

    return {
        "version": "1.0",
        "group": group,
        "generated_at": datetime.now().isoformat(),
        "defaults": {"source": "instagram", "lang": "ja", "difficulty": "medium"},
        "samples": samples,
    }


def load_existing_manifest(manifest_path: Path) -> dict | None:
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_manifest(manifest: dict, manifest_path: Path) -> None:
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")


def process_group(
    fixtures_dir: Path,
    group: str,
    rename: bool = False,
    prefix: str = "ig",
    lang: str = "ja",
    dry_run: bool = False,
) -> dict[str, Any]:
    group_dir = fixtures_dir / group
    manifest_path = group_dir / "manifest.json"
    result: dict[str, Any] = {"group": group, "path": str(group_dir), "renames": [], "samples_count": 0}

    if not group_dir.exists():
        result["error"] = f"Group directory not found: {group_dir}"
        return result

    if not dry_run:
        (group_dir / "expected").mkdir(exist_ok=True)

    if rename:
        result["renames"] = rename_files(group_dir, group=group, prefix=prefix, lang=lang, dry_run=dry_run)

    existing = load_existing_manifest(manifest_path)
    manifest = build_manifest(group_dir, group, existing)
    result["samples_count"] = len(manifest["samples"])

    if not dry_run:
        save_manifest(manifest, manifest_path)
        result["manifest_path"] = str(manifest_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile fixture manifest from image files.")
    parser.add_argument("--fixtures-dir", type=Path, default=FIXTURES_DIR)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--rename", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prefix", type=str, default="ig")
    parser.add_argument("--lang", type=str, default="ja")
    parser.add_argument("--init", type=str, default=None)
    args = parser.parse_args()

    fixtures_dir = args.fixtures_dir.resolve()
    if not fixtures_dir.exists():
        print(f"Error: Fixtures directory not found: {fixtures_dir}", file=sys.stderr)
        sys.exit(1)

    if args.init:
        new_group_dir = fixtures_dir / args.init
        if new_group_dir.exists():
            print(f"Group already exists: {args.init}")
        else:
            new_group_dir.mkdir(parents=True)
            (new_group_dir / "expected").mkdir()
            print(f"Created group: {args.init}")
        return

    groups = [args.group] if args.group else get_groups(fixtures_dir)
    if not groups:
        print("No groups found. Use --init <name> to create one.")
        sys.exit(0)

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processing {len(groups)} group(s)...\n")

    for group in groups:
        print(f"=== Group: {group} ===")
        result = process_group(
            fixtures_dir=fixtures_dir,
            group=group,
            rename=args.rename,
            prefix=args.prefix,
            lang=args.lang,
            dry_run=args.dry_run,
        )
        if "error" in result:
            print(f"  Error: {result['error']}")
            continue
        if result["renames"]:
            shown = result["renames"][:5]
            for r in shown:
                print(f"    {r['old']} -> {r['new']}")
            extra = len(result["renames"]) - len(shown)
            if extra > 0:
                print(f"    ... and {extra} more")
        print(f"  Samples: {result['samples_count']}")
        if not args.dry_run and "manifest_path" in result:
            print(f"  Manifest: {result['manifest_path']}")
        print()

    print("[DRY RUN] No changes applied." if args.dry_run else "Done!")


if __name__ == "__main__":
    main()
