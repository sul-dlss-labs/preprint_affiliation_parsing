from itertools import groupby

import pandas as pd
import spacy
import streamlit as st
from utils import analyze_blocks, get_affiliation_blocks

# Load textcat model
nlp = spacy.load("training/textcat/model-best")

# Main content area
col1, col2 = st.columns([1, 1])

# Process the text
analyzed_blocks = analyze_blocks(
    st.session_state.pdf_text, nlp, st.session_state.threshold
)
affiliation_blocks = get_affiliation_blocks(
    st.session_state.pdf_text, nlp, st.session_state.threshold
)
affiliations = " ".join([block["text"] for block in affiliation_blocks])
blocks_by_page = groupby(affiliation_blocks, key=lambda block: block["page"])
pages = [page for page, _ in blocks_by_page]

# Display the block analysis heatmap
with col1:
    st.header("Block heatmap")
    block_heatmap = []
    for page in analyzed_blocks:
        block_heatmap.append(
            [f"{'🟩' if block['is_affiliation'] else '⬜'}" for block in page]
        )
    st.text("\n".join(["".join(row) for row in block_heatmap]))

# Display the extracted affiliation text
with col2:
    st.header("Output")
    st.write(affiliations)

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
