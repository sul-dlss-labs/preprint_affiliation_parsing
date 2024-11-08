import spacy
import spacy_transformers  # required to load transformer models
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph
from utils import (
    get_affiliation_graph,
    get_affiliation_text,
    load_model,
    set_affiliation_ents,
)

# Get affiliations and run NER on them
nlp = load_model(st.session_state.ner_model)
nlp.disable_pipes("parser")
textcat = spacy.load("training/textcat/model-best")
affiliations = get_affiliation_text(st.session_state.pdf_text, textcat, 0.75)
doc = nlp(affiliations)
set_affiliation_ents(nlp, doc)
graph = get_affiliation_graph(doc)

# Display the analyzed text
col1, col2 = st.columns([2, 1])
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
            node={"labelProperty": "label"},
            link={"renderLabel": True},
        ),
    )
with col2:
    st.header("Nodes")
    nodes_md = []
    for node, node_attrs in graph.nodes(data=True):
        nodes_md.append(f"{node} _{node_attrs['type']}_")
    if nodes_md:
        st.write(f"- " + "\n- ".join(nodes_md))
    st.header("Edges")
    edges_md = []
    for u, v, edge_attrs in graph.edges(data=True):
        edges_md.append(f"{u} *{edge_attrs['type']}* {v}") 
    if edges_md:
        st.write(f"- " + "\n- ".join(edges_md))
