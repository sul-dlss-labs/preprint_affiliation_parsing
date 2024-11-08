import json
import pathlib
import random
import re
from collections import defaultdict

import networkx as nx
import spacy
import streamlit as st
from spacy.matcher import Matcher
from spacy.tokens import Span

# Preload all preprints
all_preprints = {}
all_openalex_ids = []
for file in pathlib.Path("assets/preprints/txt").glob("*.txt"):
    all_preprints[file.stem] = file.read_text(encoding="utf-8")
    all_openalex_ids.append(file.stem)


@st.cache_resource
def load_model(name: str) -> spacy.language.Language:
    """Load a spaCy model."""
    return spacy.load(name)


@st.cache_resource
def get_preprint_text(openalex_id):
    """Get the text of a preprint by its OpenAlex ID."""
    return all_preprints[openalex_id]


@st.cache_resource
def get_preprint_metadata(openalex_id):
    """Get the metadata of a preprint by its OpenAlex ID."""
    try:
        metadata = pathlib.Path(f"assets/preprints/json/{openalex_id}.json").read_text(
            encoding="utf-8"
        )
        return json.loads(metadata)
    except FileNotFoundError:
        return {}

def get_cocina_affiliations(metadata):
    """Get the affiliations from a cocina metadata object."""
    contributors = metadata.get("description", {}).get("contributor", [])
    if not contributors:
        return []

    output = []
    for contributor in contributors:
        names = contributor.get("name")
        if not names:
            continue

        roles = contributor.get("role")
        if roles:
            role_values = [role.get("value") for role in roles]
            if "author" not in role_values:
                continue

        for name in names:
            if values := name.get("structuredValue"):
                output.append(" ".join([value.get("value") for value in values]))
            if value := name.get("value"):
                output.append(value)
    return output


def is_affiliation(doc, threshold):
    """Check the textcat scores for a doc to determine if it contains affiliations."""
    return (
        doc.cats.get("AFFILIATION", 0) > threshold
        and doc.cats.get("NOT_AFFILIATION", 0) < 1 - threshold
    )


def random_preprint():
    """Select a preprint at random and store in streamlit session state."""
    st.session_state.selected_preprint = random.choice(all_openalex_ids)


def choose_preprint(openalex_id):
    """Store the selected preprint in streamlit session state."""
    st.session_state.selected_preprint = openalex_id


def get_affiliation_text(
    text: str,  # The text to search for affiliations
    nlp: spacy.language.Language,  # The spaCy model to use for text classification
    threshold: float,  # The minimum probability for a block to be considered an affiliation
) -> str:
    return " ".join(
        [block["text"] for block in get_affiliation_blocks(text, nlp, threshold)]
    )


def get_affiliation_blocks(
    text: str,  # The text to search for affiliations
    nlp: spacy.language.Language,  # The spaCy model to use for text classification
    threshold: float,  # The minimum probability for a block to be considered an affiliation
) -> list[dict]:
    """Extract and combine likely affiliation blocks from a given text."""
    # 1. Analyze all blocks and flatten
    page_blocks = analyze_blocks(text, nlp, threshold)
    all_blocks = [block for page in page_blocks for block in page]
    affiliation_blocks = []

    # 2. Move through blocks until we identify the first possible affiliation
    block = all_blocks.pop(0)
    while not block["is_affiliation"]:
        block = all_blocks.pop(0)

    # 4. Get all affiliations on that page
    first_affiliation_page = block["page"]
    affiliation_blocks += get_affiliation_range(page_blocks[first_affiliation_page])

    # 5. Check next page to see if first three blocks have an affiliation
    next_page = first_affiliation_page + 1
    while next_page < len(page_blocks) and any(
        block["is_affiliation"] for block in page_blocks[next_page][:3]
    ):
        affiliation_blocks += get_affiliation_range(page_blocks[next_page])
        next_page += 1

    # 6. Combine all affiliation blocks into a single string
    return affiliation_blocks


def get_affiliation_range(blocks: list[dict]) -> list[dict]:
    """Get the range of blocks between first and last affiliation."""
    # Get the first and last affiliation blocks
    all_affiliations = list(filter(lambda block: block["is_affiliation"], blocks))
    first_affiliation_block = all_affiliations[0]
    last_affiliation_block = all_affiliations[-1]

    # Return all blocks between the first and last affiliation
    return [
        block
        for block in blocks
        if block["index"] >= first_affiliation_block["index"]
        and block["index"] <= last_affiliation_block["index"]
    ]


def analyze_blocks(
    text: str,
    nlp: spacy.language.Language,
    threshold: float,
) -> list[str]:
    pages = text.split("\n\n")
    blocks = [page.split("\n") for page in pages]
    block_docs = [[nlp(block) for block in page] for page in blocks]
    return [
        [
            {
                "index": block,
                "page": page,
                "text": block_doc.text,
                "is_affiliation": is_affiliation(block_doc, threshold),
                "cats": block_doc.cats,
            }
            for block, block_doc in enumerate(page_docs)
        ]
        for page, page_docs in enumerate(block_docs)
    ]


# Define the pattern for matching affiliation keys
KEY_PATTERN = r"^[a-z*†‡§¶#]$|^\d{1,3}$"
KEYS_PATTERN = [
    {"TEXT": {"REGEX": KEY_PATTERN}, "ENT_TYPE": "", "OP": "+"},
]


def set_affiliation_ents(nlp, doc):
    """
    Filter entities to only include those that are relevant for affiliations,
    and add affiliation keys.
    """
    # Drop unused ent categories
    new_ents = [ent for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]
    doc.ents = new_ents

    # If any entities might include a key at the beginning, adjust their start
    new_ents = doc.ents
    for ent in new_ents:
        if ent.start > 0 and re.match(KEY_PATTERN, doc[ent.start].text):
            ent.start += 1
    doc.ents = new_ents

    # Add affiliation keys
    add_affiliation_keys(nlp, doc)

    # Return the modified doc
    return doc


def get_affiliation_keys(nlp, doc):
    """
    Return tokens in a doc that match the affiliation key pattern.
    Expects to have been run after the NER component.
    """
    matcher = Matcher(nlp.vocab)
    matcher.add("KEYS", [KEYS_PATTERN])
    matches = matcher(doc)

    # Create a text:tokens mapping of potential keys
    key_map = defaultdict(list)
    for _id, start, end in matches:
        if len(doc[start:end]) == 1:
            token = doc[start]
            key_map[token.text].append(token)

    # Keep only keys that occur at least twice
    keys = []
    for _key_text, tokens in key_map.items():
        if len(tokens) >= 2:
            keys.extend(tokens)

    # Drop keys that are the wrong part of speech (e.g. determiner "a")
    keys = [key for key in keys if key.pos_ in ["NUM", "NOUN", "PROPN", "PUNCT"]]

    return keys


def add_affiliation_keys(nlp, doc):
    """
    Adds affiliation keys to a spaCy doc.
    Expects to have been run after the NER component.
    """
    keys = get_affiliation_keys(nlp, doc)
    for key in keys:
        span = Span(doc, key.i, key.i + 1, label="KEY")
        try:
            doc.ents = list(doc.ents) + [span]
        except ValueError:  # potential key already part of a different entity
            pass


class NonKeyedAffiliationParser:
    """Parser for affiliations where each author is followed by their affiliation."""

    _graph: nx.DiGraph
    _current_person: Span
    _current_affiliation: list[Span]

    def parse_doc(self, doc):
        """Parse a spaCy doc for affiliations."""
        self._graph = nx.DiGraph()
        self._current_person = None
        self._current_affiliation = []
        for previous_ent, current_ent in zip(doc.ents, doc.ents[1:]):
            self._process_ent(current_ent, previous_ent)
        return self._graph

    def _process_ent(self, current_ent, previous_ent):
        # call more specific function based on ent type, if any
        match current_ent.label_:
            case "PERSON":
                self._process_person(current_ent, previous_ent)
            case "ORG":
                self._process_org(current_ent, previous_ent)
            case "GPE":
                self._process_gpe(current_ent, previous_ent)

    def _process_person(self, person, _previous_ent):
        # emit nodes for the person and affiliation and connect them
        if self._current_affiliation and self._current_person:
            self._emit_relationship(self._current_person, self._current_affiliation)
            self._current_affiliation = []
        self._current_person = person

    def _process_org(self, affiliation, previous_ent):
        # add the org to the current affiliation
        if self._current_person:
            if previous_ent.label_ == "ORG":
                self._current_affiliation.append(affiliation)

    def _process_gpe(self, gpe, _previous_ent):
        # add the gpe to the current affiliation
        if self._current_person:
            if self._current_affiliation:
                self._current_affiliation.append(gpe)

    def _emit_relationship(self, person, affiliation):
        person_node = self._emit_person(person)
        affiliation_head_node = self._emit_affiliation(affiliation)
        self._graph.add_edge(person_node, affiliation_head_node, type="affiliated with")

    def _emit_person(self, person):
        self._graph.add_node(
            person.text,
            label=person.text,
            span=person,
            type="person",
        )
        return person.text

    def _emit_affiliation(self, affiliation):
        last_node = None
        last_node_type = None

        # Create nodes for each part of the affiliation
        for i in range(1, len(affiliation) + 1):
            parts = affiliation[-i:]
            node_id = ", ".join([part.text for part in parts])
            if node_id not in self._graph:
                self._graph.add_node(
                    node_id,
                    label=parts[0].text,
                    span=affiliation,
                    type=parts[0].label_.lower(),
                )
            if last_node:
                if last_node_type == "ORG" and parts[0].label_ == "ORG":
                    self._graph.add_edge(node_id, last_node, type="part of")
                else:
                    self._graph.add_edge(node_id, last_node, type="located in")
            last_node = node_id
            last_node_type = parts[0].label_

        # Return the last node created (most specific)
        return last_node


class KeyedAffiliationParser:
    """Parser for affiliations where keys link authors to affiliations."""

    _graph: nx.DiGraph
    _keys: dict[list[Span]]

    def parse_doc(self, doc):
        """Parse a spaCy doc for affiliations."""
        self._graph = nx.DiGraph()
        self._keys = {
            key: [] for key in [t.text for t in doc.ents if t.label_ == "KEY"]
        }
        self._org_keys = {}
        self._parse_affiliations(doc)
        self._parse_authors(doc)
        self._emit_relationships()
        return self._graph

    def _get_first_affiliation_index(self, doc) -> int:
        """Get the point in the doc where authors end and affiliations start."""
        # Heuristic: each org has exactly one key preceding it
        # Find the first org in the doc, back up one token, and return that index
        for ent in doc.ents:
            if ent.label_ == "ORG":
                return ent.start - 1
        raise ValueError("No affiliations found in document.")
    
    def _emit_relationships(self) -> None:
        """Create relationships between authors and affiliations."""
        for key, contents in self._keys.items():
            org = self._org_keys.get(key)
            if org:
                for person in contents:
                    self._graph.add_edge(person.text, org, type="affiliated with")

    def _parse_authors(self, doc) -> None:
        """Parse the authors in the document."""
        end = self._get_first_affiliation_index(doc)
        current_person = None
        for ent in doc.ents:
            if ent.start >= end:
                break
            if ent.label_ == "PERSON":
                if ent.text not in self._graph:
                    self._graph.add_node(
                        ent.text,
                        label=ent.text,
                        span=ent,
                        type="person",
                    )
                current_person = ent
            if ent.label_ == "KEY":
                if current_person:
                    self._keys[ent.text].append(current_person)

    def _parse_affiliations(self, doc) -> None:
        """Parse the affiliations in the document."""
        start = self._get_first_affiliation_index(doc)
        current_key = None
        current_affiliation = []
        for previous_ent, current_ent in zip(doc.ents, doc.ents[1:]):
            if current_ent.start < start:
                continue
            match current_ent.label_:
                case "PERSON":
                    if current_key and current_affiliation:
                        self._emit_affiliation(current_key, current_affiliation)
                    current_key = None
                    current_affiliation = []
                case "KEY":
                    if current_key and current_affiliation:
                        self._emit_affiliation(current_key, current_affiliation)
                    current_key = current_ent.text
                    current_affiliation = []
                case "ORG":
                    if previous_ent.label_ in ["KEY", "ORG"]:
                        current_affiliation.append(current_ent)
                case "GPE":
                    if previous_ent.label_ in ["ORG", "GPE"]:
                        current_affiliation.append(current_ent)
        if current_key and current_affiliation:
            self._emit_affiliation(current_key, current_affiliation)

    def _emit_affiliation(self, key, affiliation):
        last_node = None
        last_node_type = None

        # Create nodes for each part of the affiliation
        for i in range(1, len(affiliation) + 1):
            parts = affiliation[-i:]
            node_id = ", ".join([part.text for part in parts])
            if node_id not in self._graph:
                self._graph.add_node(
                    node_id,
                    label=parts[0].text,
                    span=affiliation,
                    type=parts[0].label_.lower(),
                )
            if last_node:
                if last_node_type == "ORG" and parts[0].label_ == "ORG":
                    self._graph.add_edge(node_id, last_node, type="part of")
                else:
                    self._graph.add_edge(node_id, last_node, type="located in")
            last_node = node_id
            last_node_type = parts[0].label_

        # Return the last node created (most specific)
        self._org_keys[key] = last_node
        return last_node


def get_affiliation_graph(doc) -> nx.graph:
    """Create a graph from the affiliations in a doc."""
    if any(ent.label_ == "KEY" for ent in doc.ents):
        parser = KeyedAffiliationParser()
    else:
        parser = NonKeyedAffiliationParser()
    return parser.parse_doc(doc)
    # TODO: prune any nodes without edges?
