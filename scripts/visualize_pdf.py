import streamlit as st
from streamlit_dimensions import st_dimensions
from streamlit_pdf_viewer import pdf_viewer
from utils import get_cocina_affiliations

# Columns
col1, col2 = st.columns([2, 1])

# Preview the PDF
with col1:
  st.subheader("PDF Preview")
  width = st_dimensions()["width"]
  pdf_viewer(st.session_state.pdf_path, width=width, pages_to_render=[1])

# Preview the cocina
with col2:
  st.subheader("Cocina Authors")
  st.write(get_cocina_affiliations(st.session_state.pdf_meta))
