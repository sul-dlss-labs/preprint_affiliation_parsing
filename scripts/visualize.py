import pathlib

import spacy_streamlit
import spacy_transformers  # required to load transformer models
import streamlit as st

if __name__ == "__main__":
    # Better display for longer texts
    st.set_page_config(layout="wide")

    # Preload curated (trimmed) preprints
    curated_preprints = []
    for file in pathlib.Path("datasets/curated").glob("*.txt"):
        curated_preprints.append(
            {
                "text": file.read_text(),
                "meta": {
                    "openalex_id": file.stem,
                },
            }
        )

    # Handler to get the preprint text from its ID
    def get_preprint(openalex_id):
        return next(
            preprint
            for preprint in curated_preprints
            if preprint["meta"]["openalex_id"] == openalex_id
        )

    # Dropdown to select a preprint
    text = st.selectbox(
        "Select a preprint",
        options=[preprint["meta"]["openalex_id"] for preprint in curated_preprints],
    )

    # Display the analyzed text with option to select a model
    spacy_streamlit.visualize(
        ["training/model-best", "en_core_web_lg", "en_core_web_trf"],
        get_default_text=lambda _lang: get_preprint(text)["text"],
        visualizers=["ner"],
        ner_labels=["PERSON", "ORG", "GPE"],
    )
