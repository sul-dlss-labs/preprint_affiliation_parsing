import streamlit as st
from utils import (
                   all_openalex_ids,
                   choose_preprint,
                   get_preprint_metadata,
                   get_preprint_text,
                   load_model,
                   random_preprint,
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

# Parameter settings
st.sidebar.header("Parameters", divider=True)
st.sidebar.toggle("Remove line numbers", value=True, key="remove_numbered_lines")
st.sidebar.toggle("Space after punctuation", value=True, key="space_after_punct")
st.sidebar.toggle("Collapse whitespace", value=True, key="collapse_whitespace")
st.sidebar.toggle("Fix diacritics", value=True, key="fix_diacritics")
st.sidebar.number_input(
    "Threshold for affiliation classification",
    min_value=0.5,
    max_value=1.0,
    value=0.7,
    step=0.05,
    key="threshold",
)
st.sidebar.selectbox(
    "NER Model",
    model_names,
    key="ner_model",
)
model_load_state = st.info(f"Loading model '{st.session_state.ner_model}'...")
nlp = load_model(st.session_state.ner_model)
nlp.disable_pipes("parser")
model_load_state.empty()
st.sidebar.subheader("Pipeline info")
desc = f"""<p style="font-size: 0.85em; line-height: 1.5"><strong>{st.session_state.ner_model}:</strong> <code>v{nlp.meta['version']}</code>. {nlp.meta.get("description", "")}</p>"""
st.sidebar.markdown(desc, unsafe_allow_html=True)

# Set the path to selected PDF and its text and metadata
st.session_state.pdf_path = (
    f"assets/preprints/pdf/{st.session_state.selected_preprint}.pdf"
)
st.session_state.pdf_text = get_preprint_text(st.session_state.selected_preprint)
st.session_state.pdf_meta = get_preprint_metadata(st.session_state.selected_preprint)

# Info
st.markdown(
    """
    # Affiliation Visualization
    This application visualizes the affiliation extraction and parsing process. Choose a page in the sidebar to get started! You can select a preprint from the dropdown menu or use the quick-select buttons for some examples.
    """
)

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
