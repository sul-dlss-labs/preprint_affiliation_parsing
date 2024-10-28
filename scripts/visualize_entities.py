import spacy
import spacy_streamlit
import spacy_transformers  # required to load transformer models
import streamlit as st
from utils import get_affiliations, load_model, set_affiliation_ents

# Load model and run it
textcat = spacy.load("training/textcat/model-best")
affiliations = get_affiliations(st.session_state.pdf_text, textcat, 0.75)
nlp = load_model(st.session_state.ner_model)
if nlp.has_pipe("merge_entities"):
    nlp.disable_pipe("merge_entities")
doc = nlp(affiliations)
set_affiliation_ents(nlp, doc)

# Main content area
spacy_streamlit.visualize_ner(doc, labels=["PERSON", "ORG", "GPE", "KEY"], show_table=False)
spacy_streamlit.visualize_tokens(doc)
