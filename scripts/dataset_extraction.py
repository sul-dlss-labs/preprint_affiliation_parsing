#!/usr/bin/env python

import pathlib
from itertools import islice

import jsonlines
import spacy
import typer
from rich import print
from rich.progress import track


def slice_every(n, iterable):
    """Yield slices of size n from the input iterable."""
    iterator = iter(iterable)
    slice = list(islice(iterator, n))
    while slice:
        yield slice
        slice = list(islice(iterator, n))


def main(input_dir: pathlib.Path, output_file: pathlib.Path) -> None:
    """Generate a dataset for training the affiliation extraction model."""
    # Delete the output file if it already exists
    if output_file.exists():
        output_file.unlink()

    # Parse all plaintext files in the input directory
    # Map of OpenAlex ID: contents
    texts = {text.stem: text.read_text() for text in input_dir.glob("*.txt")}

    # Load spaCy model used to segment text into sentences
    nlp = spacy.load("en_core_web_lg")

    # Segment each file into sentences; write each "slice" of sentences as
    # a separate record in the output file along with some metadata
    created_docs = 0
    with jsonlines.open(output_file.resolve(), mode="w") as writer:
        for openalex_id, text in track(
            texts.items(), description="Creating affiliation extraction dataset..."
        ):
            doc = nlp(text)
            for slice_id, slice in enumerate(slice_every(10, doc.sents)):
                writer.write(
                    {
                        "text": " ".join(sent.text for sent in slice),
                        "meta": {
                            "openalex_id": openalex_id,
                            "slice": slice_id,
                        },
                    }
                )
                created_docs += 1
    print(f"Created dataset with {created_docs} docs.")


if __name__ == "__main__":
    typer.run(main)

__doc__ = main.__doc__
