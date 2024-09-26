#!/usr/bin/env python

import pathlib
import re
import unicodedata

import pymupdf
import pypdf
import typer
from rich import print
from rich.progress import track


def extract_text_pymupdf(path) -> str:
    doc = pymupdf.open(path)
    return "\n".join(page.get_text() for page in doc)


def extract_text_pypdf(path) -> str:
    reader = pypdf.PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages)


def remove_numbered_lines(text: str) -> str:
    lines = text.splitlines()
    return "\n".join([line for line in lines if not re.match(r"^\d+\W*$", line)])


def normalize_whitespace(text) -> str:
    contents = " ".join(text.strip().split())
    return re.sub(r"\s{2,}", " ", contents)


def normalize_unicode(text) -> str:
    return unicodedata.normalize("NFKC", text)


def normalize(text) -> str:
    return normalize_unicode(normalize_whitespace(remove_numbered_lines(text)))


def main(input_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    """Extract text from all PDFs, normalize, and output to text files."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear text files in output directory
    for txt_path in output_dir.glob("*.txt"):
        txt_path.unlink()

    # Extract text from each PDF and write to text file in output directory
    pdf_paths = list(sorted(input_dir.glob("**/*.pdf")))
    total = 0
    for pdf_path in track(pdf_paths, description="Extracting text..."):
        output_path = pathlib.Path(output_dir, f"{pdf_path.stem}.txt")
        try:
            output_path.write_text(normalize(extract_text_pymupdf(pdf_path)))
            total += 1
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")

    print(f"Extracted text from {total} PDFs.")


if __name__ == "__main__":
    typer.run(main)

__doc__ = main.__doc__
