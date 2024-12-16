import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

# Display the analyzed text
col1, col2 = st.columns([2, 1])
with col1:
    st.header("Graph")
    agraph_nodes = [
        Node(id=node, label=node_attrs["label"], type=node_attrs["type"])
        for node, node_attrs in st.session_state.affiliation_graph.nodes(data=True)
    ]
    agraph_edges = [
        Edge(source=u, target=v, label=edge_attrs["type"])
        for u, v, edge_attrs in st.session_state.affiliation_graph.edges(data=True)
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
    st.header("Output")
    st.write(st.session_state.affiliation_dict)
    st.header("Nodes")
    nodes_md = []
    for node, node_attrs in st.session_state.affiliation_graph.nodes(data=True):
        nodes_md.append(f"{node} _{node_attrs['type']}_")
    if nodes_md:
        st.write(f"- " + "\n- ".join(nodes_md))
    st.header("Edges")
    edges_md = []
    for u, v, edge_attrs in st.session_state.affiliation_graph.edges(data=True):
        edges_md.append(f"{u} *{edge_attrs['type']}* {v}") 
    if edges_md:
        st.write(f"- " + "\n- ".join(edges_md))
