import pandas as pd
import spacy
import streamlit as st
from streamlit_dimensions import st_dimensions
from streamlit_pdf_viewer import pdf_viewer
from utils import (
    all_openalex_ids,
    choose_preprint,
    get_affiliations,
    get_preprint_text,
    is_affiliation,
    random_preprint,
)

if __name__ == "__main__":
    # Better display for longer texts
    st.set_page_config(layout="wide")

    # Load textcat model
    nlp = spacy.load("training/extract/model-best")

    # Columns
    col1, col2, col3 = st.columns([3, 1, 1])

    # Dropdown to select a preprint
    with col1:
        openalex_id = st.selectbox(
            "Select a preprint",
            options=all_openalex_ids,
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
    st.button(
        "🗒️ Affiliations as footnotes", on_click=lambda: choose_preprint("W3124742002")
    )

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

    # Display the first page of the PDF
    with col5:
        width = st_dimensions()["width"]
        pdf_viewer(f"assets/preprints/pdf/{openalex_id}.pdf", width=width, pages_to_render=[1])
