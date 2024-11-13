import inspect

import networkx as nx
import spacy
import spacy_transformers  # noqa: F401
from clean_preprints import pdf_bytes_to_struct, text_from_struct
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field
from utils import (get_affiliation_graph, get_affiliation_pairs,
                   get_affiliation_text, set_affiliation_ents)

ner_model = None
textcat_model = None


# memoized helpers for loading models so we don't have to reload them on every request
def load_ner_model():
    global ner_model
    if ner_model is None:
        ner_model = spacy.load("en_core_web_trf")
    return ner_model


def load_textcat_model():
    global textcat_model
    if textcat_model is None:
        textcat_model = spacy.load("training/textcat/model-best")
    return textcat_model


# Helper to run the entire processing pipeline on an uploaded file
async def analyze_pdf(file, textcat, ner, threshold=0.75) -> nx.Graph:
    if inspect.iscoroutinefunction(file.read):
        pdf_bytes = await file.read()
    else:
        pdf_bytes = file.read()
    pdf_struct = pdf_bytes_to_struct(pdf_bytes)
    pdf_text = text_from_struct(pdf_struct)
    affiliation_text = get_affiliation_text(pdf_text, textcat, threshold)
    doc = ner(affiliation_text)
    doc = set_affiliation_ents(ner, doc)
    return get_affiliation_graph(doc)


## API schema

app = FastAPI()


class Organization(BaseModel):
    name: str = Field(description="The name of the organization")


class Person(BaseModel):
    name: str = Field(description="The full name of the person")
    affiliation: Organization | None = Field(
        None, description="Organization the person is affiliated with, if any"
    )


class Document(BaseModel):
    authors: list[Person] = Field(
        [], description="List of authors and their affiliations"
    )


@app.post("/analyze")
async def analyze_file(file: UploadFile) -> Document:
    """
    Analyze a PDF file uploaded as form data.

    Usage:
    ```
    curl -F file=@assets/preprints/pdf/W2901173781.pdf "http://localhost:8000/analyze"
    ```
    """
    # Validate that we received a PDF (very shallowly)
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be in PDF format")

    # Load the models
    textcat = load_textcat_model()
    ner = load_ner_model()

    # Analyze the document
    graph = await analyze_pdf(file, textcat, ner)

    # Format all of the authors & affiliations
    people = []
    for person_name, org_name in get_affiliation_pairs(graph):
        people.append(Person(name=person_name, affiliation=Organization(name=org_name)))

    # Return as JSON
    return {
        "authors": people,
    }
