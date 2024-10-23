import spacy
import spacy_streamlit
import spacy_transformers  # required to load transformer models
import streamlit as st
from streamlit_dimensions import st_dimensions
from streamlit_pdf_viewer import pdf_viewer
from utils import (
    add_affiliation_keys,
    all_openalex_ids,
    choose_preprint,
    get_affiliation_keys,
    get_affiliations,
    get_preprint_text,
    load_model,
    random_preprint,
)

if __name__ == "__main__":
    # Better display for longer texts
    st.set_page_config(layout="wide")

    # Title
    st.title("Named entities in PDFs")

    # Selectable models
    model_names = ["en_core_web_lg", "en_core_web_trf"]

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
        st.button("🙂 Simple", on_click=lambda: choose_preprint("W3183339884"))
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
        st.button("🇦 Superscripts", on_click=lambda: choose_preprint("W4383550744"))

        # Pipeline selector toggles
        spacy_model = st.sidebar.selectbox(
            "NER Model",
            model_names,
            key="ner_model",
        )
        model_load_state = st.info(f"Loading model '{spacy_model}'...")
        nlp = load_model(spacy_model)
        nlp.disable_pipes("parser")
        model_load_state.empty()

        st.subheader("Pipeline info")
        desc = f"""<p style="font-size: 0.85em; line-height: 1.5"><strong>{spacy_model}:</strong> <code>v{nlp.meta['version']}</code>. {nlp.meta.get("description", "")}</p>"""
        st.sidebar.markdown(desc, unsafe_allow_html=True)

    # Main content area
    col1, col2 = st.columns([1, 1])

    # Set the path to selected PDF and its text
    pdf_path = f"assets/preprints/pdf/{openalex_id}.pdf"

    # Get affiliations and run NER on them
    text = get_preprint_text(openalex_id)
    textcat = spacy.load("training/extract/model-best")
    affiliations = get_affiliations(text, textcat, 0.75)
    doc = nlp(affiliations)

    # Remove unused NER tags and add affiliation keys
    new_ents = [ent for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]
    doc.ents = new_ents
    keys = get_affiliation_keys(nlp, doc)
    add_affiliation_keys(nlp, doc)
    

    # Display the analyzed text
    with col1:
        spacy_streamlit.visualize_ner(doc, labels=["PERSON", "ORG", "GPE", "KEY"])
        st.header("Keys")
        st.write(keys)
        spacy_streamlit.visualize_tokens(doc)

    # Display the first page of the PDF
    with col2:
        width = st_dimensions()["width"]
        pdf_viewer(pdf_path, width=width, pages_to_render=[1])
