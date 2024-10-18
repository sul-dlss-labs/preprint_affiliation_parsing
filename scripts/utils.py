import pathlib
import random

import spacy
import streamlit as st

# Preload all preprints
all_preprints = {}
all_openalex_ids = []
for file in pathlib.Path("assets/preprints/txt").glob("*.txt"):
    all_preprints[file.stem] = file.read_text(encoding="utf-8")
    all_openalex_ids.append(file.stem)


@st.cache_resource
def load_model(name: str) -> spacy.language.Language:
    """Load a spaCy model."""
    return spacy.load(name)


@st.cache_resource
def get_preprint_text(openalex_id):
    """Get the text of a preprint by its OpenAlex ID."""
    return all_preprints[openalex_id]


def is_affiliation(doc, threshold):
    """Check the textcat scores for a doc to determine if it contains affiliations."""
    return (
        doc.cats.get("AFFILIATION", 0) > threshold
        and doc.cats.get("NOT_AFFILIATION", 0) < 1 - threshold
    )


def random_preprint():
    """Select a preprint at random and store in streamlit session state."""
    st.session_state.selected_preprint = random.choice(all_openalex_ids)


def choose_preprint(openalex_id):
    """Store the selected preprint in streamlit session state."""
    st.session_state.selected_preprint = openalex_id


def get_affiliations(
    text: str,  # The text to search for affiliations
    nlp: spacy.language.Language,  # The spaCy model to use for text classification
    window: int,  # The number of initial blocks to search
    threshold: float,  # The minimum probability for a block to be considered an affiliation
) -> str:
    """Extract and combine likely affiliation blocks from a given text."""
    blocks = text.split("\n")
    block_docs = [nlp(block) for block in blocks]
    return [
        block_doc.text
        for block_doc in block_docs[:window]
        if is_affiliation(block_doc, threshold)
    ]


def analyze_blocks(
    text: str,
    nlp: spacy.language.Language,
    threshold: float,
) -> list[str]:
    pages = text.split("\n\n")
    blocks = [page.split("\n") for page in pages]
    block_docs = [[nlp(block) for block in page] for page in blocks]
    return [
        [
            {
                "text": block_doc.text,
                "is_affiliation": is_affiliation(block_doc, threshold),
            }
            for block_doc in page_docs
        ]
        for page_docs in block_docs
    ]
