#!/usr/bin/env python3
"""
Evaluationsskript für den IMDb LLM-Scraper.

Berechnet Precision, Recall und F1-Score pro Feld,
verglichen mit manuell verifizierten Ground-Truth-Daten.

Verwendung:
    python evaluation/evaluate.py output/imdb_top_X_TIMESTAMP.json
"""
import json
import sys
import re
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"

# Felder und ihre Vergleichsstrategie
FELDER = {
    "titel":            "string_fuzzy",
    "jahr":             "exact",
    "bewertung":        "numeric_tolerance",  # ±0.15 akzeptiert
    "dauer_minuten":    "exact",
    "regisseur":        "string_fuzzy",
    "genres":           "set_overlap",
}


def normalize_string(s: str) -> str:
    """Normalisiert einen String für den Vergleich."""
    if not s:
        return ""
    return re.sub(r'\s+', ' ', s.strip().lower())


def is_match(field: str, predicted: Any, expected: Any) -> bool:
    """Prüft ob ein extrahierter Wert mit dem Ground Truth übereinstimmt."""
    strategy = FELDER.get(field, "exact")

    if predicted is None or expected is None:
        return predicted == expected

    if strategy == "exact":
        return predicted == expected

    elif strategy == "string_fuzzy":
        # Normalisiert und prüft ob expected in predicted vorkommt
        pred = normalize_string(str(predicted))
        exp = normalize_string(str(expected))
        return exp in pred or pred in exp

    elif strategy == "numeric_tolerance":
        try:
            return abs(float(predicted) - float(expected)) <= 0.15
        except (ValueError, TypeError):
            return False

    elif strategy == "set_overlap":
        # Für Genres: F1 über Mengenüberschneidung
        pred_set = {normalize_string(g) for g in (predicted or [])}
        exp_set = {normalize_string(g) for g in (expected or [])}
        if not exp_set:
            return True
        overlap = pred_set & exp_set
        precision = len(overlap) / len(pred_set) if pred_set else 0
        recall = len(overlap) / len(exp_set) if exp_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1 >= 0.5  # mind. 50% Überschneidung = korrekt

    return predicted == expected


def evaluate(scraper_output_path: str) -> dict:
    """
    Führt die Evaluation durch.

    Returns:
        Dict mit Metriken pro Feld und gesamt
    """
    # Ground Truth laden
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)
    ground_truth = {film["imdb_id"]: film for film in gt_data["filme"]}

    # Scraper-Output laden
    with open(scraper_output_path, encoding="utf-8") as f:
        scraper_data = json.load(f)
    scraper_films = {m["imdb_id"]: m for m in scraper_data["movies"] if m.get("imdb_id")}

    # Gemeinsame Filme
    common_ids = set(ground_truth.keys()) & set(scraper_films.keys())
    only_in_gt = set(ground_truth.keys()) - set(scraper_films.keys())

    console.print(f"\n[bold cyan]Evaluation: {Path(scraper_output_path).name}[/bold cyan]")
    console.print(f"Ground Truth Filme: {len(ground_truth)}")
    console.print(f"Scraper Filme:      {len(scraper_films)}")
    console.print(f"Gemeinsame Filme:   {len(common_ids)}")
    if only_in_gt:
        console.print(f"[yellow]Nicht extrahiert:   {len(only_in_gt)} "
                      f"({', '.join(sorted(only_in_gt)[:3])}...)[/yellow]")

    # Metriken pro Feld berechnen
    field_stats = {field: {"tp": 0, "fp": 0, "fn": 0, "none_pred": 0} for field in FELDER}

    for imdb_id in common_ids:
        gt = ground_truth[imdb_id]
        pred = scraper_films[imdb_id]

        for field in FELDER:
            expected = gt.get(field)
            predicted = pred.get(field)

            if predicted is None:
                field_stats[field]["none_pred"] += 1
                if expected is not None:
                    field_stats[field]["fn"] += 1
            elif is_match(field, predicted, expected):
                field_stats[field]["tp"] += 1
            else:
                field_stats[field]["fp"] += 1
                if expected is not None:
                    field_stats[field]["fn"] += 1

    # Tabelle ausgeben
    table = Table(title="\nEvaluationsergebnisse pro Feld", box=box.ROUNDED)
    table.add_column("Feld", style="cyan", width=18)
    table.add_column("Precision", justify="right", width=10)
    table.add_column("Recall", justify="right", width=10)
    table.add_column("F1-Score", justify="right", width=10)
    table.add_column("None-Rate", justify="right", width=10)
    table.add_column("Korrekt/Gesamt", justify="right", width=16)

    total_tp = total_fp = total_fn = 0

    results = {}
    for field, stats in field_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        none_rate = stats["none_pred"] / len(common_ids) if common_ids else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        total_tp += tp
        total_fp += fp
        total_fn += fn

        color = "green" if f1 >= 0.8 else ("yellow" if f1 >= 0.5 else "red")
        table.add_row(
            field,
            f"{precision:.2f}",
            f"{recall:.2f}",
            f"[{color}]{f1:.2f}[/{color}]",
            f"{none_rate:.0%}",
            f"{tp}/{len(common_ids)}"
        )
        results[field] = {"precision": precision, "recall": recall, "f1": f1}

    # Gesamt-Zeile
    total_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * total_p * total_r / (total_p + total_r) if (total_p + total_r) > 0 else 0
    table.add_row("─" * 18, "─" * 10, "─" * 10, "─" * 10, "─" * 10, "─" * 16)
    color = "green" if total_f1 >= 0.8 else ("yellow" if total_f1 >= 0.5 else "red")
    table.add_row(
        "[bold]GESAMT[/bold]",
        f"[bold]{total_p:.2f}[/bold]",
        f"[bold]{total_r:.2f}[/bold]",
        f"[bold][{color}]{total_f1:.2f}[/{color}][/bold]",
        "",
        f"[bold]{total_tp}/{len(common_ids) * len(FELDER)}[/bold]"
    )

    console.print(table)

    # Coverage (wie viele GT-Filme wurden überhaupt extrahiert?)
    coverage = len(common_ids) / len(ground_truth) if ground_truth else 0
    console.print(f"\n[bold]Coverage:[/bold] {coverage:.0%} "
                  f"({len(common_ids)}/{len(ground_truth)} Filme extrahiert)")

    return {
        "felder": results,
        "gesamt": {"precision": total_p, "recall": total_r, "f1": total_f1},
        "coverage": coverage,
        "n_filme": len(common_ids),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[red]Verwendung: python evaluation/evaluate.py <scraper_output.json>[/red]")
        console.print("Beispiel:   python evaluation/evaluate.py output/imdb_top_20_*.json")
        sys.exit(1)

    evaluate(sys.argv[1])
