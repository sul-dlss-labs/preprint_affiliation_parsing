# !/usr/bin/env python

import contextlib
import pathlib
import re
import unicodedata

import pypdf
import typer
from rich import print
from rich.progress import track


def extract_text_from_pdf(path) -> str:
    reader = pypdf.PdfReader(path)
    return " ".join([page.extract_text() for page in reader.pages])


def normalize_whitespace(text) -> str:
    contents = " ".join(text.strip().split())
    return re.sub(r"\s{2,}", " ", contents)


def normalize_unicode(text) -> str:
    return unicodedata.normalize("NFKC", text)


def normalize(text) -> str:
    return normalize_unicode(normalize_whitespace(text))


def main(input_dir: pathlib.Path, output_dir: pathlib.Path):
    """Extract text from all PDFs, normalize, and output to text files."""
    # Clear text files in output directory
    for txt_path in output_dir.glob("*.txt"):
        txt_path.unlink()

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract text from each PDF and write to text file in output directory
    pdf_paths = list(sorted(input_dir.glob("*.pdf")))
    total = 0
    for pdf_path in track(pdf_paths, description="Extracting text..."):
        output_path = pathlib.Path(output_dir, f"{pdf_path.stem}.txt")

        # Suppress warnings from pypdf
        with contextlib.redirect_stderr(None):
            contents = extract_text_from_pdf(pdf_path)

        # If text extraction didn't yield anything, warn and move on
        normalized = normalize(contents)
        if not normalized:
            print(f"[red]Failed to extract text from {pdf_path}.[/red]")
            continue

        # Write normalized text to output file
        output_path.write_text(normalize(contents))
        total += 1

    print(f"Extracted text from {total} PDFs.")


if __name__ == "__main__":
    typer.run(main)

__doc__ = main.__doc__
