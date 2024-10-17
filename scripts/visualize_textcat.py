import pathlib
import random

import pandas as pd
import spacy
import streamlit as st
from create_parsing_dataset import get_affiliations, is_affiliation
from streamlit_pdf_viewer import pdf_viewer

if __name__ == "__main__":
    # Better display for longer texts
    st.set_page_config(layout="wide")

    # Load textcat model
    nlp = spacy.load("training/extract/model-best")

    # Preload all preprints
    all_preprints = []
    all_openalex_ids = []
    for file in pathlib.Path("assets/preprints/txt").glob("*.txt"):
        all_preprints.append(
            {
                "text": file.read_text(),
                "meta": {
                    "openalex_id": file.stem,
                },
            }
        )
        all_openalex_ids.append(file.stem)

    # Get the preprint text from its ID
    def get_preprint_text(openalex_id):
        return next(
            preprint["text"]
            for preprint in all_preprints
            if preprint["meta"]["openalex_id"] == openalex_id
        )

    # Handler to get a random preprint
    def random_preprint():
        st.session_state.selected_preprint = random.choice(all_openalex_ids)

    # Handler to choose a preprint
    def choose_preprint(openalex_id):
        st.session_state.selected_preprint = openalex_id

    # Columns
    col1, col2, col3 = st.columns([3, 1, 1])

    # Dropdown to select a preprint
    with col1:
        openalex_id = st.selectbox(
            "Select a preprint",
            options=[preprint["meta"]["openalex_id"] for preprint in all_preprints],
            key="selected_preprint",
        )

    # Number input for selecting threshold
    with col2:
        threshold = st.number_input(
            "Threshold for affiliation classification",
            min_value=0.5,
            max_value=1.0,
            value=0.75,
            step=0.05,
        )

    # Number input for selecting window
    with col3:
        window = st.number_input(
            "Minimum number of initial chunks to search",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
        )

    # Quick-select buttons for some preprints + randomizer
    st.button("🔄 Random preprint", type="primary", on_click=random_preprint)
    st.button("#️⃣ Line numbers", on_click=lambda: choose_preprint("W4226140866"))
    st.button("🗒️ Affiliations as footnotes", on_click=lambda: choose_preprint("W3124742002"))

    # Display the selected preprint
    st.title(openalex_id)

    # More columns
    col4, col5 = st.columns([1, 1])

    # Display the extracted affiliation text
    with col4:
        text = get_preprint_text(openalex_id)
        st.header("Extracted affiliations")
        affiliations = get_affiliations(text, nlp, window, threshold)
        st.markdown(f"> {' '.join(affiliations)}")

        # Display the breakdown of the analyzed chunks
        chunks = text.split("\n")[:window]
        chunk_docs = [nlp(chunk) for chunk in chunks]
        df = pd.DataFrame(
            [
                [
                    doc.text,
                    is_affiliation(doc, threshold),
                    doc.cats["AFFILIATION"],
                    doc.cats["NOT_AFFILIATION"],
                ]
                for doc in chunk_docs
            ],
            columns=["text", "decision", "AFFILIATION", "NOT_AFFILIATION"],
        )
        st.header("Chunk breakdown")
        st.dataframe(df, use_container_width=True)

    # Display the pdf
    with col5:
        pdf_viewer(f"assets/preprints/pdf/{openalex_id}.pdf", pages_to_render=[1])
