from __future__ import annotations

import sys
from pathlib import Path

import click

from .context import AppContext
from .pipeline import discover_images, process_image


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
def digest(path: Path, mode: str, smart: bool, outputs: tuple[str, ...]) -> None:
    """Process one image or a directory of images.

    Prints OCR text and classification to stdout.
    Low-confidence results are written to review_queue.json.
    """
    AppContext.instance()  # validate config before processing begins

    images = discover_images(path)
    if not images:
        click.echo(f"No images found at: {path}", err=True)
        sys.exit(1)

    for img_path in images:
        click.echo(f"--- {img_path.name} ---")
        result = process_image(
            image_path=img_path,
            mode=mode,
            smart=smart,
            outputs=outputs,
        )
        if result is None:
            click.echo("  [skipped — unprocessable]")
            continue

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

        click.echo(f"  ocr_text    :")
        for line in result["ocr_text"].splitlines():
            click.echo(f"    {line}")
