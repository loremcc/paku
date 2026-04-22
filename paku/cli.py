from __future__ import annotations

import sys
from pathlib import Path

import click

from .context import AppContext
from .pipeline import BatchReport, discover_images, process_batch, process_image


def _print_batch_report(report: BatchReport, report_flag: bool) -> None:
    click.echo("")
    click.echo("=== Batch Summary ===")
    click.echo(f"  Total images   : {report.total}")
    click.echo(f"  Processed      : {report.processed}")
    click.echo(f"  Skipped        : {report.skipped}  (checkpoint)")
    click.echo(f"  Failed         : {report.failed}")
    click.echo(f"  Review queued  : {report.review_queued}")
    if report_flag and report.by_content_type:
        click.echo("  By content type:")
        for ct, count in sorted(report.by_content_type.items()):
            click.echo(f"    {ct:12s}: {count}")


@click.group()
@click.version_option(package_name="paku")
def cli() -> None:
    """paku — Instagram screenshot data pipeline."""


@cli.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--mode",
    type=click.Choice(["auto", "anime", "url", "recipe"]),
    default="auto",
    show_default=True,
    help="Extraction mode.",
)
@click.option(
    "--smart",
    is_flag=True,
    default=False,
    help="Enable LangExtract (LLM) path.",
)
@click.option(
    "--output",
    "outputs",
    multiple=True,
    type=click.Choice(["txt", "json", "csv", "notion"]),
    help="Output formats (repeatable).",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="Skip images already processed (checkpoint file).",
)
@click.option(
    "--report",
    is_flag=True,
    default=False,
    help="Print detailed breakdown by content type at end of batch.",
)
def digest(
    path: Path,
    mode: str,
    smart: bool,
    outputs: tuple[str, ...],
    resume: bool,
    report: bool,
) -> None:
    """Process one image or a directory of images.

    For a single image: prints OCR text and classification to stdout.
    For a directory: runs batch mode with progress bar and summary.
    Low-confidence results are written to review_queue.json.
    """
    AppContext.instance()  # validate config before processing begins

    if path.is_dir():
        _run_batch(path, mode, smart, list(outputs), resume, report)
    else:
        _run_single(path, mode, smart, list(outputs))


def _run_single(path: Path, mode: str, smart: bool, outputs: list[str]) -> None:
    click.echo(f"--- {path.name} ---")
    result = process_image(
        image_path=path,
        mode=mode,
        smart=smart,
        outputs=outputs,
    )
    if result is None:
        click.echo("  [skipped — unprocessable]")
        return

    click.echo(f"  screen_type : {result['screen_type']}")
    click.echo(f"  content_type: {result['content_type']}")
    click.echo(f"  engine      : {result['engine']}")

    extraction = result.get("extraction")
    if extraction:
        click.echo(f"  url         : {extraction.get('resolved_url', '—')}")
        click.echo(f"  confidence  : {extraction.get('confidence', '—')}")
        click.echo(f"  tier        : {extraction.get('extraction_tier', '—')}")
        if extraction.get("needs_review"):
            click.echo("  ** needs review **")
    elif result.get("status") != "pending_extraction":
        click.echo(f"  status      : {result.get('status', '—')}")

    click.echo("  ocr_text    :")
    for line in result["ocr_text"].splitlines():
        safe_line = line.encode("ascii", errors="replace").decode("ascii")
        click.echo(f"    {safe_line}")


def _run_batch(
    path: Path,
    mode: str,
    smart: bool,
    outputs: list[str],
    resume: bool,
    report_flag: bool,
) -> None:
    images = discover_images(path)
    if not images:
        click.echo(f"No images found at: {path}", err=True)
        sys.exit(1)

    click.echo(f"Found {len(images)} image(s) in {path}")

    progress_state: dict[str, int] = {"bar_value": 0}

    with click.progressbar(
        length=len(images),
        label="Processing",
        show_pos=True,
    ) as bar:
        def _callback(current: int, total: int, name: str) -> None:
            advance = current - progress_state["bar_value"]
            if advance > 0:
                bar.update(advance)
                progress_state["bar_value"] = current

        batch_report, results = process_batch(
            root=path,
            mode=mode,
            smart=smart,
            outputs=outputs,
            resume=resume,
            progress_callback=_callback,
        )
        # Advance bar to completion for processed images
        remaining = len(images) - progress_state["bar_value"]
        if remaining > 0:
            bar.update(remaining)

    _post_batch_outputs(results, mode, outputs, path)
    _print_batch_report(batch_report, report_flag)


def _post_batch_outputs(
    results: list[dict],
    mode: str,
    outputs: list[str],
    root: Path,
) -> None:
    from .context import AppContext

    ctx = AppContext.instance()
    output_dir = Path(ctx.config.get("outputs", {}).get("base_dir", "./output"))

    if "txt" in outputs:
        _write_consolidated_txt(results, output_dir)

    if "csv" in outputs and mode == "anime":
        _write_anime_csv(results, output_dir)


def _write_consolidated_txt(results: list[dict], output_dir: Path) -> None:
    from .outputs.txt_out import write_batch_txt

    by_type: dict[str, list[str]] = {}
    for r in results:
        ct = r.get("content_type", "unknown")
        extraction = r.get("extraction") or {}
        if ct == "anime":
            extras = r.get("extractions") or [extraction]
            for ex in extras:
                val = ex.get("canonical_title") or ex.get("raw_title")
                if val:
                    by_type.setdefault(ct, []).append(val)
            continue
        elif ct == "url":
            value = extraction.get("resolved_url")
        elif ct == "recipe":
            value = extraction.get("title")
        else:
            value = None
        if value:
            by_type.setdefault(ct, []).append(value)

    name_map = {"anime": "anime_titles.txt", "url": "urls.txt", "recipe": "recipe_titles.txt"}
    for ct, entries in by_type.items():
        filename = name_map.get(ct, f"{ct}_titles.txt")
        out = write_batch_txt(entries, output_dir / filename)
        click.echo(f"  [txt] {out}")


def _write_anime_csv(results: list[dict], output_dir: Path) -> None:
    from .models import AnimeExtractionResult
    from .outputs.csv_out import write_anime_csv

    anime_results: list[AnimeExtractionResult] = []
    for r in results:
        if r.get("content_type") == "anime":
            extras = r.get("extractions") or ([r["extraction"]] if r.get("extraction") else [])
            for ex in extras:
                try:
                    anime_results.append(AnimeExtractionResult.model_validate(ex))
                except Exception:
                    pass

    if not anime_results:
        return

    out = write_anime_csv(anime_results, output_dir / "anime_export.csv")
    click.echo(f"  [csv] {out}")
