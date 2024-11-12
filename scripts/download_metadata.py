#!/usr/bin/env python

import csv
import json
import pathlib

import requests
import typer
from rich import print
from rich.progress import track

# Column names in the input spreadsheet
ID_COLUMN = "OpenAlex ID"
DRUID_COLUMN = "DRUID"

# Base path for stacks and purl
PURL_BASE = "https://sul-purl-stage.stanford.edu"


def download_cocina(druid: str, output_path: pathlib.Path) -> None:
    """Download cocina JSON for a given druid."""
    cocina_url = f"{PURL_BASE}/{druid.removeprefix('druid:')}.json"
    response = requests.get(
        cocina_url, allow_redirects=True, headers={"Accept": "application/json"}
    )
    response.raise_for_status()
    output_path.write_text(json.dumps(response.json(), indent=2, ensure_ascii=False))


def main(input_file: pathlib.Path, output_dir: pathlib.Path) -> None:
    """Download cocina metadata for all objects in input spreadsheet to output directory."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear json files in output directory
    for json_path in output_dir.glob("*.json"):
        json_path.unlink()

    # Download each file listed in the PDF URL column of the input spreadsheet
    downloaded = 0
    with open(input_file, "r") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        for row in track(rows, description="Downloading cocina...", total=len(rows)):
            druid = row[DRUID_COLUMN]
            openalex_id = row[ID_COLUMN].removeprefix("https://openalex.org/")
            cocina_name = f"{openalex_id}.json"
            cocina_path = pathlib.Path(output_dir, cocina_name)
            try:
                download_cocina(druid, cocina_path)
                downloaded += 1
            except Exception as e:
                print(f"Error downloading '{druid}': {e}")
    print(f"Downloaded metadata for {downloaded} PDFs.")


if __name__ == "__main__":
    typer.run(main)

__doc__ = main.__doc__
