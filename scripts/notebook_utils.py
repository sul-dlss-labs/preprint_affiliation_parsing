import json
import os
import pathlib

from utils import get_cocina_affiliations

root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PROJECT_ROOT = pathlib.Path(root)
results_path = PROJECT_ROOT / 'results'

# notebook function for fetching preprint text
def get_preprint_text(preprint_id):
    fp = PROJECT_ROOT / "assets" / "preprints" / "txt" / f"{preprint_id}.txt"
    try:
        return fp.read_text(encoding='utf-8')
    except FileNotFoundError:
        print(f"Preprint text not found for {preprint_id}")
        return ""

# notebook function for fetching gold affiliations from cocina
def get_gold_affiliations(preprint_id):
    fp = PROJECT_ROOT / "assets" / "preprints" / "json" / f"{preprint_id}.json"
    try:
        json_str = fp.read_text(encoding='utf-8')
        cocina = json.loads(json_str)
        return get_cocina_affiliations(cocina)
    except FileNotFoundError:
        print(f"Cocina data not found for {preprint_id}")
        return ""

# notebook function for fetching pre-saved predictions
def load_predictions():
    prediction_files = list(results_path.glob("*.json"))
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
