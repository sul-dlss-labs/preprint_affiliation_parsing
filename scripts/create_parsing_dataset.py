#!/usr/bin/env python

import pathlib

import jsonlines
import spacy
import typer
from rich import print
from rich.progress import track
from utils import all_preprints, get_affiliations


def main(
    output_file: pathlib.Path,
    threshold: float = 10,
    window: float = 0.2,
) -> None:
    """Generate a dataset for training the affiliation parsing model."""
    # Delete the output file if it already exists
    if output_file.exists():
        output_file.unlink()

    # Load spaCy model used to detect affiliations
    nlp = spacy.load("training/extract/model-best")

    # Extract possible affiliation chunks from each file
    created_docs = 0
    print(
        f"Keeping chunks with >{threshold} affiliation probability in first {window} chunks."
    )
    with jsonlines.open(output_file.resolve(), mode="w") as writer:
        for openalex_id, text in track(
            all_preprints.items(), description="Creating dataset..."
        ):
            writer.write(
                {
                    "text": "\n".join(get_affiliations(text, nlp, threshold, window)),
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