import streamlit as st

# from create_extraction_dataset import get_target_lines
from clean_preprints import (
    clean_pdf_struct,
    collapse_lines,
    collapse_spans,
    collapse_whitespace,
    fix_diacritics_struct,
    pdf_to_struct,
    remove_numbered_lines,
    space_after_punct,
)
from streamlit_dimensions import st_dimensions
from streamlit_pdf_viewer import pdf_viewer
from utils import all_openalex_ids, choose_preprint, random_preprint


# Add "page 1", "block 1", etc. labels to the PDF structure for display
def annotate_pdf_struct(pdf_struct):
    annotated_struct = {}
    for page_num, page in enumerate(pdf_struct):
        annotated_page = {}
        for block_num, block in enumerate(page):
            annotated_block = {}
            for line_num, line in enumerate(block):
                annotated_line = {}
                for span_num, span in enumerate(line):
                    annotated_line[f"span {span_num}"] = span
                annotated_block[f"line {line_num}"] = annotated_line
            annotated_page[f"block {block_num}"] = annotated_block
        annotated_struct[f"page {page_num}"] = annotated_page
    return annotated_struct


if __name__ == "__main__":
    # Better display for longer texts
    st.set_page_config(layout="wide")

    # Title
    st.title("Text extraction from PDFs")

    # Sidebar
    with st.sidebar:
        # Dropdown to select a preprint
        openalex_id = st.selectbox(
            "Select a preprint",
            options=all_openalex_ids,
            key="selected_preprint",
        )

        # Quick-select buttons for some preprints + randomizer
        st.button("🔄 Random preprint", type="primary", on_click=random_preprint)
        st.header("Examples")
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

        # Toggles for normalization functions
        st.header("Normalization")
        st.toggle("Remove line numbers", value=True, key="remove_numbered_lines")
        st.toggle("Space after punctuation", value=True, key="space_after_punct")
        st.toggle("Collapse whitespace", value=True, key="collapse_whitespace")
        st.toggle("Fix diacritics", value=True, key="fix_diacritics")

        # Display toggles
        st.header("Display")
        st.toggle("Show raw output", value=False, key="show_raw_output")

    # Main content area
    col1, col2 = st.columns([1, 1])

    # Set the path to selected PDF
    pdf_path = f"assets/preprints/pdf/{openalex_id}.pdf"

    # Convert PDF to structured data
    pdf_struct = pdf_to_struct(pdf_path)
    annotated_pdf_struct = annotate_pdf_struct(pdf_struct)
    first_page_struct = {"page 0": annotated_pdf_struct["page 0"]}

    # Get the toggled-on normalization functions to apply
    norm_fns = []
    if st.session_state.remove_numbered_lines:
        norm_fns.append(remove_numbered_lines)
    norm_fns.append(collapse_spans)
    if st.session_state.space_after_punct:
        norm_fns.append(space_after_punct)
    norm_fns.append(collapse_lines)
    if st.session_state.collapse_whitespace:
        norm_fns.append(collapse_whitespace)
    if st.session_state.fix_diacritics:
        norm_fns.append(fix_diacritics_struct)

    # Clean PDF according to desired normalization functions
    result = clean_pdf_struct(pdf_struct, norm_fns)

    # Collapse blocks into strings unless the block structure is requested
    result_pages = []
    for page in result:
        result_page = "\n\n".join(page)
        result_pages.append(result_page)

    with col1:
        # Display the cleaned output
        st.header("Output")
        with st.container(height=450):
            if st.session_state.show_raw_output:
                st.write(result[0])
            else:
                st.write(result_pages[0])

        # Display the PDF block structure
        st.header("Input structure")
        with st.container(height=450):
            st.write(first_page_struct)

    # Display the first page of the PDF
    with col2:
        width = st_dimensions()["width"]
        pdf_viewer(pdf_path, width=width, pages_to_render=[1])
