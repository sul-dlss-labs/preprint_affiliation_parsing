#!/usr/bin/env python

import pathlib

import jsonlines
import spacy
import typer
from rich import print
from rich.progress import track


def get_affiliations(
    text: str,  # The text to search for affiliations
    nlp: spacy.language,  # The spaCy model to use for text classification
    window: int,  # The number of initial lines to search
    threshold: float,  # The minimum probability for a line to be considered an affiliation
) -> str:
    """Extract and combine likely affiliations from a given text."""
    lines = text.split("\n")
    line_docs = [nlp(line) for line in lines]
    return "\n".join(
        line_doc.text
        for line_doc in line_docs[:window]
        if line_doc.cats.get("AFFILIATION", 0) > threshold
        and line_doc.cats.get("NOT_AFFILIATION", 0) < 1 - threshold
    )


def main(
    input_dir: pathlib.Path,
    output_file: pathlib.Path,
    threshold: float = 10,
    window: float = 0.2,
) -> None:
    """Generate a dataset for training the affiliation parsing model."""
    # Delete the output file if it already exists
    if output_file.exists():
        output_file.unlink()

    # Parse all plaintext files in the input directory
    # Map of OpenAlex ID: contents
    texts = {text.stem: text.read_text() for text in input_dir.glob("*.txt")}

    # Load spaCy model used to detect affiliations
    nlp = spacy.load("training/extract/model-best")

    # Extract possible affiliation chunks from each file
    created_docs = 0
    print(
        f"Keeping chunks with >{threshold} affiliation probability in first {window} chunks."
    )
    with jsonlines.open(output_file.resolve(), mode="w") as writer:
        for openalex_id, text in track(
            texts.items(), description="Creating dataset..."
        ):
            writer.write(
                {
                    "text": get_affiliations(text, nlp, threshold, window),
                    "meta": {
                        "openalex_id": openalex_id,
                    },
                }
            )
            created_docs += 1
    print(f"Created dataset with {created_docs} docs.")


if __name__ == "__main__":
    typer.run(main)

__doc__ = main.__doc__
