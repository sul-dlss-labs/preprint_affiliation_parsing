import pandas as pd
import spacy
import streamlit as st
from streamlit_dimensions import st_dimensions
from streamlit_pdf_viewer import pdf_viewer
from utils import (
    all_openalex_ids,
    analyze_blocks,
    choose_preprint,
    get_affiliations,
    get_preprint_text,
    is_affiliation,
    random_preprint,
)

if __name__ == "__main__":
    # Better display for longer texts
    st.set_page_config(layout="wide")

    # Title
    st.title("Affiliation extraction from preprints")

    # Load textcat model
    nlp = spacy.load("training/extract/model-best")

    # Sidebar
    with st.sidebar:
        # Dropdown to select a preprint
        st.header("Preprint")
        openalex_id = st.selectbox(
            "Select a preprint",
            options=all_openalex_ids,
            key="selected_preprint",
        )

        # Quick-select buttons for some preprints + randomizer
        st.button("🔄 Random preprint", type="primary", on_click=random_preprint)
        st.subheader("Examples")
        st.button("🙂 Simple", on_click=lambda: choose_preprint("W3183339884"))
        st.button("#️⃣ Line numbers", on_click=lambda: choose_preprint("W4226140866"))
        st.button(
            "✏️ Affiliations as footnotes",
            on_click=lambda: choose_preprint("W3124742002"),
        )
        st.button(
            "©️ Strange symbol usage", on_click=lambda: choose_preprint("W3028990183")
        )
        st.button(
            "🌐 Symbols & accents", on_click=lambda: choose_preprint("W4385511079")
        )
        st.button(
            "📃 Multi-page affiliations",
            on_click=lambda: choose_preprint("W4386513944"),
        )
        st.button("🇦 Superscripts", on_click=lambda: choose_preprint("W4383550744"))

        # Parameter settings
        st.header("Parameters")
        threshold = st.number_input(
            "Threshold for affiliation classification",
            min_value=0.5,
            max_value=1.0,
            value=0.75,
            step=0.05,
        )
        window = st.number_input(
            "Minimum number of initial blocks to search",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
        )

    # Main content area
    col1, col2 = st.columns([1, 1])

    # Set the path to selected PDF
    pdf_path = f"assets/preprints/pdf/{openalex_id}.pdf"

    # Process the text
    text = get_preprint_text(openalex_id)
    affiliations = get_affiliations(text, nlp, window, threshold)
    page_blocks = analyze_blocks(text, nlp, threshold)

    # Display the extracted affiliation text
    with col1:
        st.header("Extracted affiliations")
        st.markdown(f"> {' '.join(affiliations)}")

        # Display the block heatmap
        st.header("Block heatmap")
        block_heatmap = []
        for i, page in enumerate(page_blocks):
            block_heatmap.append([
                f"{'🟩' if block['is_affiliation'] else '⬜'}"
                for block in page
            ])
        with st.container(height=300):
            st.text("\n".join(["".join(row) for row in block_heatmap]))

        # Display the breakdown of the analyzed blocks
        blocks = text.split("\n")[:window]
        block_docs = [nlp(block) for block in blocks]
        df = pd.DataFrame(
            [
                [
                    doc.text,
                    is_affiliation(doc, threshold),
                    doc.cats["AFFILIATION"],
                    doc.cats["NOT_AFFILIATION"],
                ]
                for doc in block_docs
            ],
            columns=["text", "decision", "AFFILIATION", "NOT_AFFILIATION"],
        )
        st.header("Block breakdown")
        st.dataframe(df, use_container_width=True)

    # Display the first page of the PDF
    with col2:
        width = st_dimensions()["width"]
        pdf_viewer(pdf_path, width=width, pages_to_render=[1])
