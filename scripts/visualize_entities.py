import spacy
import spacy_streamlit
import spacy_transformers  # required to load transformer models
import streamlit as st
from utils import (
                   add_affiliation_keys,
                   get_affiliation_keys,
                   get_affiliations,
                   load_model,
)

# Load model and run it
textcat = spacy.load("training/textcat/model-best")
affiliations = get_affiliations(st.session_state.pdf_text, textcat, 0.75)
nlp = load_model(st.session_state.ner_model)
doc = nlp(affiliations)

# Remove unused NER tags and add affiliation keys
new_ents = [ent for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]
doc.ents = new_ents
keys = get_affiliation_keys(nlp, doc)
add_affiliation_keys(nlp, doc)

# Main content area
spacy_streamlit.visualize_ner(doc, labels=["PERSON", "ORG", "GPE", "KEY"])
