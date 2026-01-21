from __future__ import annotations

from pathlib import Path

import click

from cmd.export_inpn import main as export_inpn
from pipelines.ingest_csv import DEFAULT_INPUT, DEFAULT_OUTPUT, ingest_csv


@click.group()
def cli() -> None:
    """Pipeline orchestrator for Biolit."""


@cli.command("ingest-csv")
@click.option(
    "--input-path",
    type=click.Path(path_type=Path),
    default=DEFAULT_INPUT,
    help="Chemin vers le CSV d'entree (par defaut: data/raw/observations.csv)",
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT,
    help="Chemin de sortie (par defaut: data/export_biolit.csv)",
)
def ingest_csv_cmd(input_path: Path, output_path: Path) -> None:
    """Copy the source CSV into the data workspace."""
    ingest_csv(input_path=input_path, output_path=output_path)


@cli.command("export-inpn")
def export_inpn_cmd() -> None:
    """Run the existing export + dataviz pipeline."""
    export_inpn()


if __name__ == "__main__":
    cli()
