import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

# Columns
col1, col2 = st.columns([2, 1])

# Preview the PDF
with col1:
  st.subheader("PDF Preview")

  # Save current page in session state; make two buttons for advancing
  # and going back
  if "page" not in st.session_state:
    st.session_state.page = 1
  st.write(f"Page {st.session_state.page}")
  if st.session_state.page > 1:
    if st.button("Previous page"):
      st.session_state.page -= 1
  if st.button("Next page"):
    st.session_state.page += 1

  pdf_viewer(st.session_state.pdf_path, pages_to_render=[st.session_state.page])

# Preview the cocina
with col2:
  st.subheader("Cocina Affiliations")
  st.write(st.session_state.cocina_affiliations)
