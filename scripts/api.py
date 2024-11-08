from tempfile import SpooledTemporaryFile

import spacy
import spacy_transformers  # noqa: F401
from clean_preprints import pdf_bytes_to_struct, text_from_struct
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
from utils import (
    get_affiliation_graph,
    get_affiliation_pairs,
    get_affiliation_text,
    set_affiliation_ents,
)

# Load the models
ner = spacy.load("en_core_web_trf")
ner.disable_pipes("parser")
textcat = spacy.load("training/textcat/model-best")


# Helper to run the entire processing pipeline on an uploaded file
async def analyze_pdf(file: SpooledTemporaryFile):
    pdf_bytes = await file.read()
    pdf_struct = pdf_bytes_to_struct(pdf_bytes)
    pdf_text = text_from_struct(pdf_struct)
    affiliation_text = get_affiliation_text(pdf_text, textcat, 0.75)
    doc = ner(affiliation_text)
    doc = set_affiliation_ents(ner, doc)
    return get_affiliation_graph(doc)


## API schema

app = FastAPI()


class Organization(BaseModel):
    name: str


class Person(BaseModel):
    name: str
    affiliation: Organization | None = None


class Document(BaseModel):
    authors: list[Person]


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

    # Analyze the document
    graph = await analyze_pdf(file)

    # Format all of the authors & affiliations
    people = []
    for person_name, org_name in get_affiliation_pairs(graph):
        people.append(Person(name=person_name, affiliation=Organization(name=org_name)))

    # Return as JSON
    return {
        "authors": people,
    }
