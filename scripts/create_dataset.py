#!/usr/bin/env python

import pathlib
from itertools import islice

import jsonlines
import spacy
import typer
from rich import print
from rich.progress import track


def main(input_dir: pathlib.Path, output_file: pathlib.Path, n_sents: int = 20) -> None:
    """Generate a dataset for training the affiliation parsing model."""
    # Delete the output file if it already exists
    if output_file.exists():
        output_file.unlink()

    # Parse all plaintext files in the input directory
    # Map of OpenAlex ID: contents
    texts = {text.stem: text.read_text() for text in input_dir.glob("*.txt")}

    # Load spaCy model used to segment text into sentences
    nlp = spacy.load("en_core_web_lg")

    # Segment each file into sentences; take the first n_sents sentences to
    # form a doc, since affiliations are usually listed at the beginning
    created_docs = 0
    print(f"Using the first {n_sents} sentences for training.")
    with jsonlines.open(output_file.resolve(), mode="w") as writer:
        for openalex_id, text in track(
            texts.items(), description="Creating dataset..."
        ):
            full_doc = nlp(text)
            sentences = islice(full_doc.sents, n_sents)
            writer.write(
                {
                    "text": " ".join([sentence.text for sentence in sentences]),
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
