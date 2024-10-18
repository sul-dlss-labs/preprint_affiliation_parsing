#!/usr/bin/env python

import pathlib

import jsonlines
import spacy
import typer
from rich import print
from rich.progress import track
from utils import all_preprints


def get_target_blocks(
    text: str,  # The text to search for blocks
    nlp: spacy.language.Language,  # The spaCy model to use for named entity recognition
) -> list[str]:
    """From a given text, return blocks that could be useful for training."""
    #
    # 1. Search the first and last page of the text
    # 2. Limit to blocks that include at least one named entity of type ORG or PERSON
    #
    pages = text.split("\n\n")
    first_page, *_, last_page = pages
    blocks = "\n".join([first_page, last_page]).split("\n")
    block_docs = [nlp(block) for block in blocks]
    return [
        block_doc.text
        for block_doc in block_docs
        if any(ent.label_ in ["ORG", "PERSON"] for ent in block_doc.ents)
    ]


def main(
    output_file: pathlib.Path,
) -> None:
    """Generate a dataset for training the affiliation extraction model."""
    # Delete the output file if it already exists
    if output_file.exists():
        output_file.unlink()

    # Load spaCy model used to detect named entities
    nlp = spacy.load("en_core_web_trf")

    # Segment each file into sentences and keep target sentences as training data
    created_docs = 0
    with jsonlines.open(output_file.resolve(), mode="w") as writer:
        for openalex_id, text in track(
            all_preprints.items(), description="Creating dataset..."
        ):
            for block in get_target_blocks(text, nlp):
                writer.write(
                    {
                        "text": block,
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
