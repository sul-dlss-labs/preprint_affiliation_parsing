#!/usr/bin/env python

import pathlib
from itertools import islice

import jsonlines
import spacy
import typer
from rich import print
from rich.progress import track


def main(input_dir: pathlib.Path, output_file: pathlib.Path, n_lines: int = 20, min_tokens: int = 5) -> None:
    """Generate a dataset for training the affiliation extraction model."""
    # Delete the output file if it already exists
    if output_file.exists():
        output_file.unlink()

    # Parse all plaintext files in the input directory
    # Map of OpenAlex ID: contents
    texts = {text.stem: text.read_text() for text in input_dir.glob("*.txt")}

    # Load spaCy model used to detect named entities
    nlp = spacy.load("en_core_web_trf")

    # Segment each file into sentences; take the first n_sents sentences to
    # form a doc, since affiliations are usually listed at the beginning
    created_docs = 0
    print(f"Searching the first {n_lines} lines for training data.")
    print(f"Keeping possible affiliation lines with at least {min_tokens} tokens.")
    with jsonlines.open(output_file.resolve(), mode="w") as writer:
        for openalex_id, text in track(
            texts.items(), description="Creating dataset..."
        ):
            # Treat each line as a doc; keep only lines that include at least
            # one named entity of type ORG or PERSON, since that could
            # indicate an affiliation, and the line has at least min_tokens 
            # tokens, to provide better context for training
            lines = islice(text.split("\n"), n_lines)
            line_docs = [nlp(line) for line in lines]
            target_lines = [
                line_doc.text
                for line_doc in line_docs
                if any(ent.label_ in ["ORG", "PERSON"] for ent in line_doc.ents) 
                and len(line_doc) > min_tokens
            ]

            for line in target_lines:
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
