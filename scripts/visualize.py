import spacy
import spacy_transformers  # required to load transformer models
import streamlit as st
from utils import (
                   all_openalex_ids,
                   analyze_blocks,
                   choose_preprint,
                   get_affiliation_dict,
                   get_affiliation_graph,
                   get_cocina_affiliations,
                   get_preprint_metadata,
                   get_preprint_text,
                   load_model,
                   random_preprint,
                   set_affiliation_ents,
)

st.set_page_config(
    layout="wide", page_title="Affiliation Visualization", page_icon="✨"
)

# Selectable NER models
model_names = ["en_core_web_trf", "en_core_web_lg"]

# Dropdown to select a preprint
st.sidebar.selectbox(
    "Select a preprint",
    options=all_openalex_ids,
    key="selected_preprint",
)

# Quick-select buttons for some preprints + randomizer
st.sidebar.button(
    "🔄 Random preprint",
    type="primary",
    use_container_width=True,
    on_click=random_preprint,
)
examples = st.sidebar.expander("Examples", expanded=True)
examples.button(
    "🙂 Simple",
    use_container_width=True,
    on_click=lambda: choose_preprint("W3183339884"),
)
examples.button(
    "#️⃣ Line numbers",
    use_container_width=True,
    on_click=lambda: choose_preprint("W4226140866"),
)
examples.button(
    "✏️ Footnotes",
    use_container_width=True,
    on_click=lambda: choose_preprint("W3124742002"),
)
examples.button(
    "©️ Strange symbols",
    use_container_width=True,
    on_click=lambda: choose_preprint("W3028990183"),
)
examples.button(
    "🌐 Symbols & accents",
    use_container_width=True,
    on_click=lambda: choose_preprint("W4385511079"),
)
examples.button(
    "📃 Multi-page",
    use_container_width=True,
    on_click=lambda: choose_preprint("W4386513944"),
)
examples.button(
    "🇦 Superscripts",
    use_container_width=True,
    on_click=lambda: choose_preprint("W4383550744"),
)
examples.button(
    "❌ Non-keyed",
    use_container_width=True,
    on_click=lambda: choose_preprint("W3133101250"),
)
examples.button(
    "🏛️ Single institution",
    use_container_width=True,
    on_click=lambda: choose_preprint("W3046678272"),
)
examples.button(
    "🔚 Affiliations at end",
    use_container_width=True,
    on_click=lambda: choose_preprint("W3000588783"),
)

# Parameter settings
st.sidebar.header("Parameters", divider=True)
st.sidebar.number_input(
    "Threshold for affiliation classification",
    min_value=0.2,
    max_value=1.0,
    value=0.5,
    step=0.05,
    key="threshold",
)
st.sidebar.selectbox(
    "NER Model",
    model_names,
    key="ner_model",
)
model_load_state = st.info(f"Loading model '{st.session_state.ner_model}'...")
_ner = load_model(st.session_state.ner_model)
_textcat = spacy.load("training/textcat_multilabel/model-best")
_ner.disable_pipes("parser")
model_load_state.empty()
st.sidebar.subheader("Pipeline info")
desc = f"""<p style="font-size: 0.85em; line-height: 1.5"><strong>{st.session_state.ner_model}:</strong> <code>v{_ner.meta['version']}</code>. {_ner.meta.get("description", "")}</p>"""
st.sidebar.markdown(desc, unsafe_allow_html=True)

# Info
st.markdown(
    """
    # Affiliation Visualization
    This application visualizes the affiliation extraction and parsing process. Choose a page in the sidebar to get started! You can select a preprint from the dropdown menu or use the quick-select buttons for some examples.
    """
)

# Pre-run and cache all analysis
st.session_state.pdf_path = (
    f"assets/preprints/pdf/{st.session_state.selected_preprint}.pdf"
)
st.session_state.pdf_text = get_preprint_text(st.session_state.selected_preprint)
st.session_state.pdf_meta = get_preprint_metadata(st.session_state.selected_preprint)
st.session_state.cocina_affiliations = get_cocina_affiliations(st.session_state.pdf_meta)

# Do the analysis
st.session_state.analyzed_blocks = analyze_blocks(
    st.session_state.pdf_text, _textcat, st.session_state.threshold, _ner
)
st.session_state.flat_blocks = [block for page in st.session_state.analyzed_blocks for block in page]
st.session_state.affiliation_blocks = [block for block in st.session_state.flat_blocks if block["is_affiliation"]]
st.session_state.affiliations = " ".join([block["text"] for block in st.session_state.affiliation_blocks])
st.session_state.doc = _ner(st.session_state.affiliations)
set_affiliation_ents(_ner, st.session_state.doc)
st.session_state.affiliation_graph = get_affiliation_graph(st.session_state.doc)
st.session_state.affiliation_dict = get_affiliation_dict(st.session_state.affiliation_graph)

# Page navigation
pg = st.navigation(
    [
        st.Page("visualize_pdf.py", title="PDF"),
        st.Page("visualize_blocks.py", title="Blocks"),
        st.Page("visualize_textcat.py", title="Text Classification"),
        st.Page("visualize_entities.py", title="Named Entities"),
        st.Page("visualize_relations.py", title="Relationships"),
    ]
)

# Run the page
pg.run()
