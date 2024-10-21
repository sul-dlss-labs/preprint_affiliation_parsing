#!/usr/bin/env python

import csv
import pathlib

import requests
import typer
from rich import print
from rich.progress import track

# Column names in the input spreadsheet
ID_COLUMN = "OpenAlex ID"
DRUID_COLUMN = "DRUID"

# Base path for stacks
STACKS_BASE = "https://sul-stacks-stage.stanford.edu"


def download_file(url: str, output_path: pathlib.Path) -> None:
    """Download file from URL and save to output path."""
    response = requests.get(
        url, allow_redirects=True, headers={"Accept": "application/pdf"}
    )
    response.raise_for_status()
    output_path.write_bytes(response.content)


def main(input_file: pathlib.Path, output_dir: pathlib.Path) -> None:
    """Download all PDF files in input spreadsheet to output directory."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear pdf files in output directory
    for pdf_path in output_dir.glob("*.pdf"):
        pdf_path.unlink()

    # Download each file listed in the PDF URL column of the input spreadsheet
    downloaded = 0
    with open(input_file, "r") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        for row in track(rows, description="Downloading PDFs...", total=len(rows)):
            druid = row[DRUID_COLUMN]
            openalex_id = row[ID_COLUMN].lstrip("https://openalex.org/")
            pdf_url = f"{STACKS_BASE}/{druid}/{openalex_id}.pdf"
            pdf_name = f"{openalex_id}.pdf"
            pdf_path = pathlib.Path(output_dir, pdf_name)
            try:
                download_file(pdf_url, pdf_path)
                downloaded += 1
            except Exception as e:
                print(f"Error downloading {pdf_url}: {e}")
    print(f"Downloaded {downloaded} PDFs.")


if __name__ == "__main__":
    typer.run(main)

__doc__ = main.__doc__
