from __future__ import annotations

from pathlib import Path
import shutil

import click

from biolit import DATADIR

DEFAULT_INPUT = DATADIR / "raw" / "observations.csv"
DEFAULT_OUTPUT = DATADIR / "export_biolit.csv"


def ingest_csv(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Prepare the CSV for downstream pipelines by copying it into DATADIR."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"CSV introuvable: {input_path}. Placez-le dans data/raw/observations.csv"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_path, output_path)
    return output_path


@click.command()
@click.option(
    "--input-path",
    type=click.Path(path_type=Path),
    default=DEFAULT_INPUT,
    show_default=True,
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT,
    show_default=True,
)
def main(input_path: Path, output_path: Path) -> None:
    """CLI entrypoint for CSV ingestion."""
    ingest_csv(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    main()
