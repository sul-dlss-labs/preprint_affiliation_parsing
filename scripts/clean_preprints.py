#!/usr/bin/env python

import pathlib
import re
import unicodedata

import regex
import spacy
import typer
from rich import print
from rich.progress import track
from spacy_layout import spaCyLayout


def space_after_punct(text: str) -> str:
    """Add a space after punctuation."""
    # Might have numbers or symbols immediately after
    text = re.sub(r"([,;])", r"\1 ", text)
    # Commonly used to link authors to affiliations
    text = re.sub(r"([*†‡§¶])", r" \1 ", text)
    return text


def collapse_whitespace(text: str) -> str:
    """Collapse all whitespace to a single space."""
    return " ".join(text.split())


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


NORM_FNS = [
    space_after_punct,
    collapse_whitespace,
    fix_diacritics,
]


def clean_span(span: spacy.tokens.Span) -> str:
    """Normalize a spaCy Span."""
    text = span.text
    for fn in NORM_FNS:
        text = fn(text)
    return text


def doc_to_text(doc: spacy.tokens.Doc) -> str:
    """Convert a spaCy Doc to a string using layout info."""
    pages = []
    for _layout, spans in doc._.pages:
        page = []
        for span in spans:
            page.append(clean_span(span))
        pages.append("\n".join(page))
    return "\n\n".join(pages)


def main(input_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    """Extract text from all PDFs, normalize, and output to text files."""
    # Set up the layout parser
    nlp = spacy.blank("en")
    layout = spaCyLayout(nlp)

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
            doc = layout(str(pdf_path))
            output_path.write_text(doc_to_text(doc))
            total += 1
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")

    print(f"Extracted text from {total} PDFs.")


if __name__ == "__main__":
    typer.run(main)

__doc__ = main.__doc__
