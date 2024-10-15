import jsonlines
import spacy_streamlit
import spacy_transformers  # required to load transformer models
import streamlit as st

if __name__ == "__main__":
    # Better display for longer texts
    st.set_page_config(layout="wide")

    # Preload all preprints with metadata
    all_preprints = []
    with jsonlines.open("assets/preprints.jsonl") as reader:
        for obj in reader:
            all_preprints.append(obj)

    # Handler to get the preprint text from its ID
    def get_preprint(openalex_id):
        return next(
            preprint
            for preprint in all_preprints
            if preprint["meta"]["openalex_id"] == openalex_id
        )

    # Dropdown to select a preprint
    text = st.selectbox(
        "Select a preprint",
        [preprint["meta"]["openalex_id"] for preprint in all_preprints],
    )

    # Display the analyzed text with option to select a model
    spacy_streamlit.visualize(
        ["training/model-best", "en_core_web_lg", "en_core_web_trf"],
        get_default_text=lambda _lang: get_preprint(text)["text"],
        visualizers=["ner"],
        ner_labels=["PERSON", "ORG", "GPE"],
    )
