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
        st.button(" Non-keyed affiliations", on_click=lambda: choose_preprint("W3133101250"))

    # Main content area
    col1, col2 = st.columns([1, 1])

    # Set the path to selected PDF and its text
    pdf_path = f"assets/preprints/pdf/{openalex_id}.pdf"

    # Get affiliations and run NER on them
    spacy_model = "en_core_web_trf"
    nlp = load_model(spacy_model)
    nlp.disable_pipes("parser")
    text = get_preprint_text(openalex_id)
    textcat = spacy.load("training/extract/model-best")
    affiliations = get_affiliations(text, textcat, 0.6)
    doc = nlp(affiliations)

    # Remove unused NER tags and add affiliation keys
    new_ents = [ent for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]
    doc.ents = new_ents
    add_affiliation_keys(nlp, doc)
    people = [ent for ent in doc.ents if ent.label_ == "PERSON"]
    orgs = [ent for ent in doc.ents if ent.label_ == "ORG"]
    gpes = [ent for ent in doc.ents if ent.label_ == "GPE"]
    keys = [ent for ent in doc.ents if ent.label_ == "KEY"]

    # Create the nodes
    node_ids = set()
    nodes = []
    edges = []
    for person in people:
        if person.text in node_ids:
            continue
        node_ids.add(person.text)
        nodes.append(Node(
            id=person.text,
            label=person.text,
            type="person",
        ))
    for org in orgs:
        if org.text in node_ids:
            continue
        node_ids.add(org.text)
        nodes.append(Node(
            id=org.text,
            label=org.text,
            type="org",
        ))
    for gpe in gpes:
        if gpe.text in node_ids:
            continue
        node_ids.add(gpe.text)
        nodes.append(Node(
            id=gpe.text,
            label=gpe.text,
            type="gpe",
        ))

    # Edge keys
    edge_keys = {}
    for key in keys:
        edge_keys[key.text] = {
            "people": [],
            "orgs": [],
        }

    # TODO: two different languages -> two different automata: if there are
    # no KEYs; use a much simpler parser.
    # TODO: actually enumerate the states and transitions; pull stuff out
    # into a module? see: https://python-statemachine.readthedocs.io/en/latest/readme.html
    # DFA to create edges
    last_ent, *ents = list(doc.ents)
    for ent in doc.ents:
        if ent.label_ == "KEY":
            if last_ent.label_ == "PERSON":
                edge_keys[ent.text]["people"].append(last_ent.text)
        elif ent.label_ == "ORG":
            if last_ent.label_ == "KEY":
                edge_keys[last_ent.text]["orgs"].append(ent.text)
            elif last_ent.label_ == "ORG":
                edges.append(Edge(
                    source=last_ent.text,
                    target=ent.text,
                    label="part of",
                ))
        elif ent.label_ == "GPE":
            if last_ent.label_ in ["ORG", "GPE"]:
                edges.append(Edge(
                    source=last_ent.text,
                    target=ent.text,
                    label="located in",
                ))
        last_ent = ent
    for key, values in edge_keys.items():
        for person in values["people"]:
            for org in values["orgs"]:
                edges.append(Edge(
                    source=person,
                    target=org,
                    label="affiliated with",
                ))

    # TODO: remove any nodes without edges

    # Get the cocina affiliations
    metadata = get_preprint_metadata(openalex_id)
    cocina_affiliations = get_cocina_affiliations(metadata)

    # Display the analyzed text
    with col1:
        st.header("Graph")
        agraph(
            nodes=nodes,
            edges=edges,
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
        st.write([(node.id, node.type) for node in nodes])
        st.header("Cocina")
        st.write(cocina_affiliations)
        # st.write(nodes)

    # Display the first page of the PDF
    with col2:
        width = st_dimensions()["width"]
        pdf_viewer(pdf_path, width=width, pages_to_render=[1])
