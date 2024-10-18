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
    threshold: float,  # The minimum probability for a block to be considered an affiliation
) -> str:
    return " ".join(
        [block["text"] for block in get_affiliation_blocks(text, nlp, threshold)]
    )


def get_affiliation_blocks(
    text: str,  # The text to search for affiliations
    nlp: spacy.language.Language,  # The spaCy model to use for text classification
    threshold: float,  # The minimum probability for a block to be considered an affiliation
) -> list[dict]:
    """Extract and combine likely affiliation blocks from a given text."""
    # 1. Analyze all blocks and flatten
    page_blocks = analyze_blocks(text, nlp, threshold)
    all_blocks = [block for page in page_blocks for block in page]
    affiliation_blocks = []

    # 2. Move through blocks until we identify the first possible affiliation
    block = all_blocks.pop(0)
    while not block["is_affiliation"]:
        block = all_blocks.pop(0)

    # 4. Get all affiliations on that page
    first_affiliation_page = block["page"]
    affiliation_blocks += get_affiliation_range(page_blocks[first_affiliation_page])

    # 5. Check next page to see if first three blocks have an affiliation
    next_page = first_affiliation_page + 1
    while next_page < len(page_blocks) and any(
        block["is_affiliation"] for block in page_blocks[next_page][:3]
    ):
        affiliation_blocks += get_affiliation_range(page_blocks[next_page])
        next_page += 1

    # 6. Combine all affiliation blocks into a single string
    return affiliation_blocks


def get_affiliation_range(blocks: list[dict]) -> list[dict]:
    """Get the range of blocks between first and last affiliation."""
    # Get the first and last affiliation blocks
    all_affiliations = list(filter(lambda block: block["is_affiliation"], blocks))
    first_affiliation_block = all_affiliations[0]
    last_affiliation_block = all_affiliations[-1]

    # Return all blocks between the first and last affiliation
    return [
        block
        for block in blocks
        if block["index"] >= first_affiliation_block["index"]
        and block["index"] <= last_affiliation_block["index"]
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
                "index": block,
                "page": page,
                "text": block_doc.text,
                "is_affiliation": is_affiliation(block_doc, threshold),
                "cats": block_doc.cats,
            }
            for block, block_doc in enumerate(page_docs)
        ]
        for page, page_docs in enumerate(block_docs)
    ]
