#!/usr/bin/env python

import pathlib
import re
import unicodedata
from typing import Callable

import pymupdf
import regex
import typer
from rich import print
from rich.progress import track


def pdf_to_struct(path: str) -> list:
    """Extract text blocks from a PDF using PyMuPDF."""
    # Span-by-span extraction (credit to @jcoyne) in order to ensure things like
    # affiliation markers are tokenized correctly with space around them
    doc = pymupdf.open(path)
    pdf = []
    for page in doc:
        blocks = []
        dict = page.get_textpage().extractDICT(sort=True)
        for block in dict["blocks"]:
            lines = []
            for line in block["lines"]:
                spans = []
                for i, span in enumerate(line["spans"]):
                    # Indicator for superscript text; see:
                    # https://pymupdf.readthedocs.io/en/latest/recipes-text.html#how-to-analyze-font-characteristics
                    # Add a space so that it is tokenized separately
                    if span["flags"] & 2**0:
                        spans.append(f" {span["text"]}")
                    # Special case: a single lowercase letter/number at the 
                    # beginning of a line is likely superscript, but pymupdf 
                    # sometimes doesn't flag it as such; unclear why...
                    elif len(span["text"]) == 1 and span["text"].isalnum() and i == 0:
                        spans.append(f"{span['text']} ")
                    else:
                        spans.append(span["text"])
                lines.append(spans)
            blocks.append(lines)
        pdf.append(blocks)
    return pdf 


def clean_pdf_struct(pdf_struct: list, norm_fns: list[Callable[[str], str]]):
    """Apply normalization steps to a structured PDF."""
    for fn in norm_fns:
        pdf_struct = fn(pdf_struct)
    return pdf_struct


def collapse_spans(pdf_struct: list) -> list:
    """Collapse consecutive spans in a line to a single string."""
    new_struct = []
    for page in pdf_struct:
        new_page = []
        for block in page:
            new_block = []
            for line in block:
                new_block.append("".join(line))
            new_page.append(new_block)
        new_struct.append(new_page)
    return new_struct


def space_after_punct(pdf_struct: list) -> list:
    """Add a space after some punctuation marks."""
    new_struct = []
    for page in pdf_struct:
        new_page = []
        for block in page:
            new_block = []
            for line in block:
                # Might have numbers or symbols immediately after
                new_line = re.sub(r"([,;])", r"\1 ", line)
                # Commonly used to link authors to affiliations
                new_line = re.sub(r"([*†‡§¶])", r" \1 ", new_line)
                new_block.append(new_line)
            new_page.append(new_block)
        new_struct.append(new_page)
    return new_struct


def remove_numbered_lines(pdf_struct: list) -> list:
    """Remove lines that consist only of a number, from line-numbered preprints."""
    new_struct = []
    for page in pdf_struct:
        new_page = []
        for block in page:
            new_block = []
            for line in block:
                # If the line has a single span that is a number, skip it,
                # as it is likely a line number
                if len(line) == 1 and re.match(r"^\d+\W*$", line[0]):
                    continue
                new_block.append(line)
            new_page.append(new_block)
        new_struct.append(new_page)
    return new_struct


def collapse_whitespace(pdf_struct: list) -> list:
    """Collapse consecutive whitespace inside blocks."""
    new_struct = []
    for page in pdf_struct:
        new_page = []
        for block in page:
            if block.strip():
                new_page.append(" ".join(block.split()))
        new_struct.append(new_page)
    return new_struct


def collapse_lines(pdf_struct: list) -> list:
    """Collapse consecutive lines into a single block."""
    new_struct = []
    for page in pdf_struct:
        new_page = []
        for block in page:
            new_block = " ".join(block)
            if new_block:
                new_page.append(new_block)
        new_struct.append(new_page)
    return new_struct


def fix_diacritics(text: str) -> str:
    """Fix combining diacritics in PDF text."""
    # Text coming from PyMuPDF seems to have diacritics as their "modifier symbol"
    # version, which stands apart from the character it modifies.
    #
    # We first decompose via NFKD, which will turn "´" U+00B4 ACUTE ACCENT into
    # a space followed by U+0301 COMBINING ACUTE ACCENT.
    #
    # We then use a regex to remove the space and swap the combining diacritic
    # with the character it modifies, so that the modifier follows the character.
    decomposed = unicodedata.normalize("NFKD", text)
    return regex.sub(r" (\p{Mn})(\w)", r"\2\1", decomposed)


def fix_diacritics_struct(pdf_struct: list) -> list:
    """Apply fix_diacritics to a structured PDF."""
    new_struct = []
    for page in pdf_struct:
        new_page = []
        for block in page:
            new_block = fix_diacritics(block)
            new_page.append(new_block)
        new_struct.append(new_page)
    return new_struct


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
            pdf_struct = pdf_to_struct(pdf_path)
            cleaned_pdf_struct = clean_pdf_struct(
                pdf_struct,
                [
                    remove_numbered_lines,
                    collapse_spans,
                    space_after_punct,
                    collapse_lines,
                    collapse_whitespace,
                    fix_diacritics_struct,
                ],
            )
            output_txt = ""
            for page in cleaned_pdf_struct:
                for block in page:
                    output_txt += f"{block}\n"
                output_txt += "\n"
            output_path.write_text(output_txt)
            total += 1
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")

    print(f"Extracted text from {total} PDFs.")


if __name__ == "__main__":
    typer.run(main)

__doc__ = main.__doc__
