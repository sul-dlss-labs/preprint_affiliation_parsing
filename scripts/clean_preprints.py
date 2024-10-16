#!/usr/bin/env python

import pathlib
import re
import unicodedata
from functools import reduce

import pymupdf
import typer
from rich import print
from rich.progress import track


def extract_text_pymupdf(path) -> str:
    """Extract text from a PDF using PyMuPDF."""
    # Span-by-span extraction (credit to @jcoyne) in order to ensure things like
    # affiliation markers are tokenized correctly with space around them
    doc = pymupdf.open(path)
    blocks = []
    for page in doc:
        dict = page.get_textpage().extractDICT(sort=True)
        for block in dict["blocks"]:
            block_text = ""
            for line in block["lines"]:
                for span in line["spans"]:
                    block_text += (f"{normalize(span['text'])} ")
            if block_text.strip():
                blocks.append(block_text.strip())
    return "\n".join(blocks)


def remove_numbered_lines(text: str) -> str:
    """Remove lines that consist only of a number, from line-numbered preprints."""
    lines = text.splitlines()
    return "\n".join([line for line in lines if not re.match(r"^\d+\W*$", line)])


def normalize_whitespace(text) -> str:
    """Remove leading/trailing whitespace and collapse multiple spaces into one."""
    contents = " ".join(text.strip().split())
    return re.sub(r"\s{2,}", " ", contents)


def normalize_unicode(text) -> str:
    """Ensure diacritics are combined with the preceding character."""
    # TODO: fix this behavior?
    return unicodedata.normalize("NFKC", text)


def split_commas(text) -> str:
    """Ensure commas have a space after them."""
    return re.sub(r",", ", ", text)


def split_semicolons(text) -> str:
    """Ensure semicolons have a space after them."""
    return re.sub(r";", "; ", text)


def normalize(text) -> str:
    """Apply a series of normalization steps to the text."""
    return reduce(
        lambda x, f: f(x),
        [
            remove_numbered_lines,
            split_commas,
            split_semicolons,
            normalize_whitespace,
            normalize_unicode,
        ],
        text,
    )


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
            output_path.write_text(extract_text_pymupdf(pdf_path))
            total += 1
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")

    print(f"Extracted text from {total} PDFs.")


if __name__ == "__main__":
    typer.run(main)

__doc__ = main.__doc__
