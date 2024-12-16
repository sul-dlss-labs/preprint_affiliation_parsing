from itertools import groupby

import pandas as pd
import streamlit as st

# Main content area
col1, col2 = st.columns([1, 1])

# Process the text
spans_by_page = groupby(st.session_state.affiliation_blocks, key=lambda block: block["page"])
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
    elif block["like_affiliation"]:
        return "🟧"
    else:
        return "⬜"

# Display the block analysis heatmap
with col1:
    st.header("Block heatmap")
    block_heatmap = []
    for page in st.session_state.analyzed_blocks:
        block_heatmap.append([block_emoji(block) for block in page])
    st.text("\n".join(["".join(row) for row in block_heatmap]))

# Display the extracted affiliation text
with col2:
    st.header("Output")
    st.write(st.session_state.affiliations)

# Display the breakdown of the analyzed pages
all_blocks_on_affiliation_pages = []
for i, page in enumerate(st.session_state.analyzed_blocks):
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
            block["like_affiliation"],
        ]
        for block in st.session_state.flat_blocks
    ],
    columns=["page", "decision", "text", "AFFILIATION", "AUTHOR", "CITATION", "like_affiliation"],
)
st.header("Block breakdown")
st.dataframe(df, use_container_width=True, hide_index=True)
