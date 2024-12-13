from itertools import groupby

import pandas as pd
import spacy
import streamlit as st
from utils import analyze_blocks, get_affiliation_spans

# Load textcat model
nlp = spacy.load("training/textcat_multilabel/model-best")

# Main content area
col1, col2 = st.columns([1, 1])

# Process the text
analyzed_blocks = analyze_blocks(
    st.session_state.pdf_text, nlp, st.session_state.threshold
)
flat_blocks = [block for page in analyzed_blocks for block in page]
affiliation_blocks = [block for block in flat_blocks if block["is_affiliation"]]
affiliations = " ".join([block["text"] for block in affiliation_blocks])
spans_by_page = groupby(affiliation_blocks, key=lambda block: block["page"])
pages = [page for page, _ in spans_by_page]


def block_emoji(block):
    if block["cats"]["AFFILIATION"] >= st.session_state.threshold and block["cats"]["AUTHOR"] >= st.session_state.threshold:
        return "🟩"
    elif block["cats"]["AFFILIATION"] >= st.session_state.threshold:
        return "🟨"
    elif block["cats"]["AUTHOR"] >= st.session_state.threshold:
        return "🟦"
    elif block["cats"]["CITATION"] >= st.session_state.threshold:
        return "🟥"
    else:
        return "⬜"

# Display the block analysis heatmap
with col1:
    st.header("Block heatmap")
    block_heatmap = []
    for page in analyzed_blocks:
        block_heatmap.append([block_emoji(block) for block in page])
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
            round(block["cats"]["AFFILIATION"], 2),
            round(block["cats"]["AUTHOR"], 2),
            round(block["cats"]["CITATION"], 2),
        ]
        for block in flat_blocks
    ],
    columns=["page", "decision", "text", "AFFILIATION", "AUTHOR", "CITATION"],
)
st.header("Block breakdown")
st.dataframe(df, use_container_width=True, hide_index=True)
