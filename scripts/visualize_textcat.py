import pathlib

import pandas as pd
import spacy
import streamlit as st


def is_affiliation(doc, threshold):
    return (
        doc.cats["AFFILIATION"] > threshold and doc.cats["NOT_AFFILIATION"] < threshold
    )


if __name__ == "__main__":
    # Better display for longer texts
    st.set_page_config(layout="wide")

    # Load textcat model
    nlp = spacy.load("training/extract/model-best")

    # Preload all preprints
    all_preprints = []
    for file in pathlib.Path("assets/preprints/txt").glob("*.txt"):
        all_preprints.append(
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
            for preprint in all_preprints
            if preprint["meta"]["openalex_id"] == openalex_id
        )

    # Dropdown to select a preprint
    text = st.selectbox(
        "Select a preprint",
        options=[preprint["meta"]["openalex_id"] for preprint in all_preprints],
    )

    # Number input for selecting threshold
    threshold = st.number_input(
        "Threshold for affiliation classification",
        min_value=0.5,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

    # Analyze the first ten lines of the preprint
    lines = get_preprint(text)["text"].split("\n")[:10]
    docs = [nlp(line) for line in lines]

    # Display the extracted affiliation text
    st.title("Extracted affiliations")
    affiliations = [doc.text for doc in docs if is_affiliation(doc, threshold)]
    st.markdown(f"> {'\n'.join(affiliations)}")

    # Display the breakdown of the first ten lines of analyzed text
    df = pd.DataFrame(
        [
            [
                doc.text,
                is_affiliation(doc, threshold),
                doc.cats["AFFILIATION"],
                doc.cats["NOT_AFFILIATION"],
            ]
            for doc in docs
        ],
        columns=["text", "decision", "AFFILIATION", "NOT_AFFILIATION"],
    )
    st.title("Chunk breakdown")
    st.dataframe(df, use_container_width=True)
