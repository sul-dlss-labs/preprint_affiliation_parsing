import spacy
import spacy_streamlit
import spacy_transformers  # required to load transformer models
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph
from streamlit_dimensions import st_dimensions
from streamlit_pdf_viewer import pdf_viewer
from utils import (
    add_affiliation_keys,
    all_openalex_ids,
    choose_preprint,
    get_affiliation_graph,
    get_affiliations,
    get_cocina_affiliations,
    get_preprint_metadata,
    get_preprint_text,
    load_model,
    random_preprint,
)

if __name__ == "__main__":
    # Better display for longer texts
    st.set_page_config(layout="wide")

    # Title
    st.title("Affiliation relations")

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
        st.button("❌ Non-keyed affiliations", on_click=lambda: choose_preprint("W3133101250"))

    # Main content area
    col1, col2 = st.columns([1, 1])

    # Set the path to selected PDF and its text
    pdf_path = f"assets/preprints/pdf/{openalex_id}.pdf"

    # Get affiliations and run NER on them
    spacy_model = "en_core_web_trf"
    nlp = load_model(spacy_model)
    nlp.disable_pipes("parser")
    text = get_preprint_text(openalex_id)
    textcat = spacy.load("training/textcat/model-best")
    affiliations = get_affiliations(text, textcat, 0.6)
    doc = nlp(affiliations)

    # Remove unused NER tags and add affiliation keys
    new_ents = [ent for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]
    doc.ents = new_ents
    add_affiliation_keys(nlp, doc)
    graph = get_affiliation_graph(doc)

    # Get the cocina affiliations
    metadata = get_preprint_metadata(openalex_id)
    cocina_affiliations = get_cocina_affiliations(metadata)

    # Display the analyzed text
    with col1:
        st.header("Graph")
        agraph_nodes = [
            Node(id=node, label=node_attrs["label"], type=node_attrs["type"])
            for node, node_attrs in graph.nodes(data=True)
        ]
        agraph_edges = [
            Edge(source=u, target=v, label=edge_attrs["type"])
            for u, v, edge_attrs in graph.edges(data=True)
        ]
        agraph(
            nodes=agraph_nodes,
            edges=agraph_edges,
            config=Config(
                directed=True,
                hierarchical=True,
                nodeHighlightBehavior=True,
                highlightColor="#F7A7A6",
                collapsible=True,
                node={'labelProperty': 'label'},
                link={'renderLabel': True},
            ),
        )
        st.header("Nodes")
        for node, node_attrs in graph.nodes(data=True):
            st.write(f"{node} ({node_attrs['type']})")
        st.header("Edges")
        for u, v, edge_attrs in graph.edges(data=True):
            st.write(f"{u} {edge_attrs['type']} {v}")
        st.header("Cocina")
        st.write(cocina_affiliations)
        # st.write(nodes)

    # Display the first page of the PDF
    with col2:
        width = st_dimensions()["width"]
        pdf_viewer(pdf_path, width=width, pages_to_render=[1])
