import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from utils import get_cocina_affiliations

# Columns
col1, col2 = st.columns([2, 1])

# Preview the PDF
with col1:
  st.subheader("PDF Preview")
  pdf_viewer(st.session_state.pdf_path, pages_to_render=[1])

# Preview the cocina
with col2:
  st.subheader("Cocina Authors")
  st.write(get_cocina_affiliations(st.session_state.pdf_meta))
