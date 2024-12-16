import spacy_streamlit
import streamlit as st

# Main content area
spacy_streamlit.visualize_ner(st.session_state.doc, labels=["PERSON", "ORG", "GPE", "KEY"], show_table=False)
spacy_streamlit.visualize_tokens(st.session_state.doc)
