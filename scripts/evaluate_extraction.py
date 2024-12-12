#!/usr/bin/env python3

import json
import os
import statistics
import sys
import warnings
from pathlib import Path

import spacy
import typer
from rich.console import Console
from rich.table import Column, Table
from thefuzz import fuzz

root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PROJECT_ROOT = Path(root)
sys.path.insert(1, str(PROJECT_ROOT / "scripts"))

from utils import get_affiliation_spans, get_cocina_affiliations  # noqa: E402

# Preprints that we want to focus on
PROBLEM_LIST = [
    "W2941345678",
    "W2942725897",
    "W2963628635",
    "W2976790204",
    "W3116436840",
    "W3147817170",
    "W3178821884",
    "W3199353954",
    "W4226047880",
    "W4399283731",
]

EMPTY_SCORES = {pid: 0.0 for pid in PROBLEM_LIST}


def score_prediction(prediction: str, gold: dict, threshold: float = 0.75) -> float:
    """Score a single prediction against the ground truth metadata."""
    # Get all the author names and affiliation names
    author_names = list(gold.keys())
    affiliations = [aff for affs in gold.values() for aff in affs]
    all_ents = len(author_names) + len(affiliations)

    # Count 1 point for each author name and affiliation name that is found
    # within the prediction string, with the given threshold of fuzziness
    correct = 0
    threshold_int = int(threshold * 100)
    for name in author_names:
        if fuzz.token_set_ratio(name, prediction, force_ascii=True) >= threshold_int:
            correct += 1
    for aff in affiliations:
        if fuzz.token_set_ratio(aff, prediction, force_ascii=True) >= threshold_int:
            correct += 1

    # Return the ratio of correct predictions to all predictions
    return round(correct / all_ents, 2)


def highlight_gold_with_diff(prediction: str, gold: dict) -> str:
    """Highlight the differences between the prediction and the ground truth metadata."""
    output = ""
    for author, affiliations in gold.items():
        if author in prediction:
            output += f"[green]{author}[/green]\n"
        else:
            output += f"[red]{author}[/red]\n"
        for aff in affiliations:
            if aff in prediction:
                output += f"\t[green]{aff}[/green]\n"
            else:
                output += f"\t[red]{aff}[/red]\n"
    return output


def format_delta(last_score: float, current_score: float) -> str:
    """Format the difference between two scores as a string."""
    delta = current_score - last_score
    if delta > 0:
        return f"[green]+{delta:.2f}[/green]"
    elif delta < 0:
        return f"[red]{delta:.2f}[/red]"
    else:
        return "-"


def main(
    gold_path: Path = Path("assets/preprints/json"),
    preprints_path: Path = Path("assets/preprints/txt"),
    metrics_path: Path = Path("metrics"),
    threshold: float = 0.6,
) -> None:
    """Evaluate the affiliation extraction process against ground truth text files."""
    # Load all the ground truth files
    gold_metas = {}
    for file in gold_path.glob("*.json"):
        cocina = json.loads(file.read_text("utf-8"))
        gold_metas[file.stem] = get_cocina_affiliations(cocina)

    # Set up the extraction model; ignore pytorch warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    textcat = spacy.load("training/textcat_multilabel/model-best")

    # For each preprint in the problem list, get the predicted affiliation text
    pred_texts = {}
    for preprint_id in PROBLEM_LIST:
        preprint_txt = (preprints_path / f"{preprint_id}.txt").read_text("utf-8")
        pred_texts[preprint_id] = " ".join(
            get_affiliation_spans(preprint_txt.splitlines(), textcat, threshold)
        )

    # Compute the scores and averages
    scores = {
        preprint_id: score_prediction(pred_texts[preprint_id], gold_metas[preprint_id])
        for preprint_id in PROBLEM_LIST
    }
    mean_score = round(statistics.mean(scores.values()), 2)
    median_score = round(statistics.median(scores.values()), 2)
    metrics = {
        "mean": mean_score,
        "median": median_score,
        "scores": scores,
    }

    # Load last run and best run metrics if present
    last_run = metrics_path / "extraction-last.json"
    best_run = metrics_path / "extraction-best.json"
    if last_run.is_file():
        last_metrics = json.loads(last_run.read_text("utf-8"))
    else:
        last_metrics = EMPTY_SCORES
    if best_run.is_file():
        best_metrics = json.loads(best_run.read_text("utf-8"))
    else:
        best_metrics = EMPTY_SCORES

    # Save the current metrics as the last run
    last_run.write_text(json.dumps(metrics, indent=2))

    # Print overall mean score
    # If the current mean score is better than the best, save it
    console = Console()
    if mean_score > best_metrics["mean"]:
        best_run.write_text(json.dumps(metrics, indent=2))
        console.print(f"[bold green]New best mean: {mean_score}[/bold green]")
    else:
        console.print(f"Previous best mean: {best_metrics['mean']}")

    # Create a table with the best, last, and current scores and print it
    table = Table("metric", "best", "last", "current", "delta")
    table.add_row(
        "[bold]Mean[/bold]",
        str(best_metrics["mean"]),
        str(last_metrics["mean"]),
        str(mean_score),
        format_delta(last_metrics["mean"], mean_score),
    )
    table.add_row(
        "[bold]Median[/bold]",
        str(best_metrics["median"]),
        str(last_metrics["median"]),
        str(median_score),
        format_delta(last_metrics["median"], median_score),
    )

    # Sort the preprints by their current score and display
    sorted_ids = sorted(PROBLEM_LIST, key=lambda p: scores[p], reverse=True)
    for preprint_id in sorted_ids:
        table.add_row(
            preprint_id,
            str(best_metrics["scores"][preprint_id]),
            str(last_metrics["scores"][preprint_id]),
            str(scores[preprint_id]),
            format_delta(last_metrics["scores"][preprint_id], scores[preprint_id]),
        )
    console.print(table)

    # Print the visual diffs of each predicted text with the gold text
    for preprint_id in sorted_ids:
        console.print(f"[bold]{preprint_id}[/bold] ({scores[preprint_id]})")
        table = Table.grid(
            Column(header="predicted", justify="left", ratio=1),
            Column(header="actual", justify="left", ratio=1),
            expand=True,
        )
        table.add_row(
            pred_texts[preprint_id],
            highlight_gold_with_diff(pred_texts[preprint_id], gold_metas[preprint_id]),
        )
        console.print(table)


if __name__ == "__main__":
    typer.run(main)


__doc__ = main.__doc__
