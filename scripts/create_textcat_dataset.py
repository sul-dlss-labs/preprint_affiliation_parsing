#!/usr/bin/env python

import pathlib

import jsonlines
import spacy
import typer
from rich import print
from rich.progress import track
from spacy_layout import spaCyLayout
from utils import all_openalex_ids


def not_text(span: spacy.tokens.Span) -> bool:
    """Check if a span is exclusively numbers or punctuation."""
    return all((token.like_num or token.is_punct) for token in span)


def get_target_spans(doc: spacy.tokens.Doc) -> list[str]:
    """
    Take a doc analyzed with spacy_layout and return spans useful for
    training the affiliation extraction model.
    """
    # Use only spans on the first three and last three pages of the document
    pages = doc._.pages[:3] + doc._.pages[-3:]
    spans = [span for _layout, spans in pages for span in spans]

    # Discard spans that are titles, tables, headings, etc. (not body text)
    spans = [span for span in spans if span.label_ == "text"]

    # Discard spans that are exclusively numbers or punctuation (e.g. line numbers)
    spans = [span for span in spans if not not_text(span)]

    # Return as list of strings
    return [span.text for span in spans]


def get_target_chunks(
    text: str,  # The text to search for chunks
    nlp: spacy.language.Language,  # The spaCy model to use for named entity recognition
) -> list[str]:
    """From a given text, return chunks that could be useful for training."""
    # TODO: use a fixed-size chunk here with overlap so we always have context
    # check Docling chunking settings?
    # https://ds4sd.github.io/docling/examples/hybrid_chunking/
    pass


def main(
    output_file: pathlib.Path,
) -> None:
    """Generate a dataset for training the affiliation extraction model."""
    # Delete the output file if it already exists
    if output_file.exists():
        output_file.unlink()

    # Load spaCy model for layout parsing
    nlp = spacy.blank("en")
    layout = spaCyLayout(nlp)

    # Get training data from each doc and add to JSONL file
    created_docs = 0
    with jsonlines.open(output_file.resolve(), mode="w") as writer:
        for openalex_id in track(all_openalex_ids, description="Creating dataset..."):
            doc = layout(f"assets/preprints/pdf/{openalex_id}.pdf")
            for span in get_target_spans(doc):
                writer.write(
                    {
                        "text": span,
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
