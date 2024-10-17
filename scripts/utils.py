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
    nlp: spacy.language,  # The spaCy model to use for text classification
    window: int,  # The number of initial chunks to search
    threshold: float,  # The minimum probability for a chunk to be considered an affiliation
) -> str:
    """Extract and combine likely affiliation chunks from a given text."""
    chunks = text.split("\n")
    chunk_docs = [nlp(chunk) for chunk in chunks]
    return [
        chunk_doc.text
        for chunk_doc in chunk_docs[:window]
        if is_affiliation(chunk_doc, threshold)
    ]
