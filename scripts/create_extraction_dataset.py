#!/usr/bin/env python

import pathlib
from itertools import islice

import jsonlines
import spacy
import typer
from rich import print
from rich.progress import track


def get_target_lines(
    text: str,  # The text to search for lines
    nlp: spacy.language,  # The spaCy model to use for named entity recognition
    n_lines: int,  # The number of initial lines to search
    min_tokens: int,  # The minimum number of tokens in a line to keep
) -> list[str]:
    """From a given text, return lines that could be useful for training."""
    #
    # 1. Search the first n_lines lines of the text
    # 2. Limit to lines that include at least one named entity of type ORG or PERSON
    # 3. Limit again to lines with at least min_tokens tokens
    #
    # This strategy is designed to capture many likely affiliations with
    # relevant context. Anecdotally it results in about a 50% balance in
    # the number of positive and negative examples, which is ideal for training.
    #
    lines = islice(text.split("\n"), n_lines)
    line_docs = [nlp(line) for line in lines]
    return [
        line_doc.text
        for line_doc in line_docs
        if any(ent.label_ in ["ORG", "PERSON"] for ent in line_doc.ents)
        and len(line_doc) > min_tokens
    ]


def main(
    input_dir: pathlib.Path,
    output_file: pathlib.Path,
    n_lines: int = 20,
    min_tokens: int = 5,
) -> None:
    """Generate a dataset for training the affiliation extraction model."""
    # Delete the output file if it already exists
    if output_file.exists():
        output_file.unlink()

    # Parse all plaintext files in the input directory
    # Map of OpenAlex ID: contents
    texts = {text.stem: text.read_text() for text in input_dir.glob("*.txt")}

    # Load spaCy model used to detect named entities
    nlp = spacy.load("en_core_web_trf")

    # Segment each file into sentences and keep target sentences as training data
    created_docs = 0
    print(f"Searching the first {n_lines} lines for training data.")
    print(f"Keeping possible affiliation lines with at least {min_tokens} tokens.")
    with jsonlines.open(output_file.resolve(), mode="w") as writer:
        for openalex_id, text in track(
            texts.items(), description="Creating dataset..."
        ):
            for line in get_target_lines(text, nlp, n_lines, min_tokens):
                writer.write(
                    {
                        "text": line,
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
