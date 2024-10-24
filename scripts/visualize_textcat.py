from itertools import groupby

import pandas as pd
import spacy
import streamlit as st
from streamlit_dimensions import st_dimensions
from streamlit_pdf_viewer import pdf_viewer
from utils import (
    all_openalex_ids,
    analyze_blocks,
    choose_preprint,
    get_affiliation_blocks,
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
    nlp = spacy.load("training/textcat/model-best")

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
        st.button("❌ Non-keyed affiliations", on_click=lambda: choose_preprint("W3133101250"))

        # Parameter settings
        st.header("Parameters")
        threshold = st.number_input(
            "Threshold for affiliation classification",
            min_value=0.5,
            max_value=1.0,
            value=0.7,
            step=0.05,
        )

    # Main content area
    col1, col2 = st.columns([1, 1])

    # Set the path to selected PDF
    pdf_path = f"assets/preprints/pdf/{openalex_id}.pdf"

    # Process the text
    text = get_preprint_text(openalex_id)
    analyzed_blocks = analyze_blocks(text, nlp, threshold)
    affiliation_blocks = get_affiliation_blocks(text, nlp, threshold)
    affiliations = " ".join([block["text"] for block in affiliation_blocks])
    blocks_by_page = groupby(affiliation_blocks, key=lambda block: block["page"])
    pages = [page for page, _ in blocks_by_page]

    # Display the extracted affiliation text
    with col1:
        st.header("Extracted affiliations")
        with st.container(height=200):
            st.write(affiliations)

        # Display the block analysis heatmap
        st.header("Block heatmap")
        block_heatmap = []
        for page in analyzed_blocks:
            block_heatmap.append([
                f"{'🟩' if block['is_affiliation'] else '⬜'}"
                for block in page
            ])
        with st.container(height=300):
            st.text("\n".join(["".join(row) for row in block_heatmap]))

    # Display the first page of the PDF
    with col2:
        width = st_dimensions()["width"]
        with st.container(height=600):
            pdf_viewer(pdf_path, width=width, pages_to_render=[page + 1 for page in pages])

    # Display the breakdown of the analyzed pages
    all_blocks_on_affiliation_pages = []
    for i, page in enumerate(analyzed_blocks):
        if i in pages:
            all_blocks_on_affiliation_pages.extend(page)

    df = pd.DataFrame(
        [
            [
                int(block["page"]) + 1,
                block["is_affiliation"],
                block["text"],
                block["cats"]["AFFILIATION"],
                block["cats"]["NOT_AFFILIATION"],
            ]
            for block in all_blocks_on_affiliation_pages
        ],
        columns=["page", "decision", "text", "AFFILIATION", "NOT_AFFILIATION"],
    )
    st.header("Block breakdown")
    st.dataframe(df, use_container_width=True, hide_index=True)
