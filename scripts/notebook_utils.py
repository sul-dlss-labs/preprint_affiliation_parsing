import json
import os
import pathlib
import xml.etree.ElementTree as ET

import requests
from utils import get_cocina_affiliations

root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PROJECT_ROOT = pathlib.Path(root)
RESULTS_PATH = PROJECT_ROOT / "results"
GROBID_API_URL = "http://localhost:8070/api"
TEI_NAMESPACES = {"tei": "http://www.tei-c.org/ns/1.0"}


# notebook function for fetching preprint text
def get_preprint_text(preprint_id):
    fp = PROJECT_ROOT / "assets" / "preprints" / "txt" / f"{preprint_id}.txt"
    try:
        return fp.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Preprint text not found for {preprint_id}")
        return ""


# notebook function for fetching gold affiliations from cocina
def get_gold_affiliations(preprint_id):
    fp = PROJECT_ROOT / "assets" / "preprints" / "json" / f"{preprint_id}.json"
    try:
        json_str = fp.read_text(encoding="utf-8")
        cocina = json.loads(json_str)
        return get_cocina_affiliations(cocina)
    except FileNotFoundError:
        print(f"Cocina data not found for {preprint_id}")
        return ""


# notebook function for fetching pre-saved JSON predictions
def load_predictions(path):
    prediction_files = list(path.glob("*.json"))
    predictions = {}
    for prediction_file in prediction_files:
        preprint_id = prediction_file.stem
        with prediction_file.open(mode="r") as f:
            try:
                contents = json.load(f)
                predictions[preprint_id] = contents
            except json.JSONDecodeError:
                print(f"Error loading {prediction_file}")
                continue
    return predictions


# notebook function for fetching pre-saved TEI XML predictions from GROBID
def load_predictions_xml(path):
    prediction_files = list(path.glob("*.tei.xml"))
    predictions = {}
    for prediction_file in prediction_files:
        preprint_id = prediction_file.stem.removesuffix(".tei")
        with prediction_file.open(mode="r") as f:
            try:
                contents = f.read()
            except Exception as e:
                print(f"Error loading {prediction_file}", e)
                continue
            predictions[preprint_id] = tei_xml_affiliations_to_json(contents)
    return predictions

# load TEI XML from string and convert affiliations to JSON
def tei_xml_affiliations_to_json(xml_str):
    output = {}
    try:
        root = ET.fromstring(xml_str)
        header = root.find("tei:teiHeader", namespaces=TEI_NAMESPACES)
        file_desc = header.find("tei:fileDesc", namespaces=TEI_NAMESPACES)
        source_desc = file_desc.find("tei:sourceDesc", namespaces=TEI_NAMESPACES)
        bibl_struct = source_desc.find("tei:biblStruct", namespaces=TEI_NAMESPACES)
        analytic = bibl_struct.find("tei:analytic", namespaces=TEI_NAMESPACES)
        authors = analytic.findall("tei:author", namespaces=TEI_NAMESPACES)
        for author in authors:
            # name is combination of all elements in persName element
            name = " ".join(
                [
                    el.text
                    for el in author.findall(
                        "tei:persName/*", namespaces=TEI_NAMESPACES
                    )
                ]
            )

            # affiliations are the text of the orgName element with type="institution" in each affiliation element
            affiliations = [
                el.text
                for el in author.findall(
                    "tei:affiliation/tei:orgName[@type='institution']",
                    namespaces=TEI_NAMESPACES,
                )
            ]

            # remove duplicates
            affiliations = list(set(affiliations))

            # store the name and affiliations in the output dictionary
            output[name] = affiliations
    except Exception as e:
        print(f"Error parsing TEI XML: {e}")
    return output


# send a PDF to GROBID's API and store the saved prediction TEI XML
def get_grobid_prediction(preprint_file: pathlib.Path, results_path: pathlib.Path):
    try:
        files = {"input": preprint_file.open("rb")}
    except FileNotFoundError:
        print(f"Preprint file not found: {preprint_file}")
        return None
    response = requests.post(
        f"{GROBID_API_URL}/processHeaderDocument",
        files=files,
        headers={"Accept": "application/xml"},
    )
    if response.status_code == 200:
        tei_path = results_path / f"{preprint_file.stem}.tei.xml"
        with tei_path.open("wb") as f:
            f.write(response.content)
        return tei_path
    else:
        print(f"Error processing {preprint_file} with GROBID: {response.text}")
        return None
