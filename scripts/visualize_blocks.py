import streamlit as st
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


# Main content area
col1, col2 = st.columns([1, 1])

# Convert PDF to structured data
pdf_struct = pdf_to_struct(st.session_state.pdf_path)
annotated_pdf_struct = annotate_pdf_struct(pdf_struct)
first_page_struct = {"page 0": annotated_pdf_struct["page 0"]}

# Get the toggled-on normalization functions to apply
norm_fns = []
norm_fns.append(remove_numbered_lines)
norm_fns.append(collapse_spans)
norm_fns.append(space_after_punct)
norm_fns.append(collapse_lines)
norm_fns.append(collapse_whitespace)
norm_fns.append(fix_diacritics_struct)

# Clean PDF according to desired normalization functions
result = clean_pdf_struct(pdf_struct, norm_fns)

# Collapse blocks into strings unless the block structure is requested
result_pages = []
for page in result:
    result_page = "\n\n".join(page)
    result_pages.append(result_page)

with col1:
    # Display the PDF block structure
    st.header("Input")
    st.write(first_page_struct)

with col2:
    # Display the cleaned output
    st.header("Output")
    st.write(result_pages[0])
