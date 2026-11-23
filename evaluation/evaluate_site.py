#!/usr/bin/env python3
"""
Generischer Evaluator für beliebige SiteConfig-basierte Extraktion.
Vergleicht LLM-Output mit Ground Truth via F1, Precision, Recall.

Verwendung:
    python evaluation/evaluate_site.py quotes output/quotes_*.json
    python evaluation/evaluate_site.py books  output/books_*.json
"""
import json
import re
import sys
from pathlib import Path
from typing import Any, List, Dict

from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).parent.parent))

from sites.site_config import SiteConfig, FieldConfig

console = Console()


def normalize_string(s: str) -> str:
    if not s:
        return ""
    s = str(s)
    # Typografische Anführungszeichen → ASCII
    s = s.replace('\u201c', '"').replace('\u201d', '"')
    s = s.replace('\u2018', "'").replace('\u2019', "'")
    return re.sub(r'\s+', ' ', s.strip().lower())


def is_match(field: FieldConfig, predicted: Any, expected: Any) -> bool:
    """Feldtyp-spezifischer Vergleich."""
    if predicted is None and expected is None:
        return True
    if predicted is None or expected is None:
        return False

    if field.type == "string":
        pred = normalize_string(predicted)
        exp = normalize_string(expected)
        return exp in pred or pred in exp

    elif field.type == "int":
        try:
            return int(predicted) == int(expected)
        except (ValueError, TypeError):
            return False

    elif field.type == "float":
        try:
            return abs(float(predicted) - float(expected)) <= 0.05
        except (ValueError, TypeError):
            return False

    elif field.type == "bool":
        return bool(predicted) == bool(expected)

    elif field.type == "array":
        pred_set = {normalize_string(v) for v in (predicted or [])}
        exp_set = {normalize_string(v) for v in (expected or [])}
        if not exp_set and not pred_set:
            return True
        if not exp_set:
            return True
        overlap = pred_set & exp_set
        precision = len(overlap) / len(pred_set) if pred_set else 0
        recall = len(overlap) / len(exp_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1 >= 0.5

    return predicted == expected


def evaluate_site(config: SiteConfig, results_path: str) -> dict:
    """
    Evaluiert Scraper-Output gegen Ground Truth.

    Args:
        config: SiteConfig mit ground_truth_path
        results_path: Pfad zur JSON-Ausgabe von toscrape_compare.py

    Returns:
        Dict mit Metriken pro Feld und gesamt
    """
    # Ground Truth laden (Format: {"items": [...]} oder plain list)
    gt_path = Path(__file__).parent.parent / config.ground_truth_path
    with open(gt_path, encoding="utf-8") as f:
        gt_raw = json.load(f)
    ground_truth: List[Dict] = gt_raw["items"] if isinstance(gt_raw, dict) else gt_raw

    # Ergebnisse laden (unterstützt direkte Liste oder {"results": [...]})
    with open(results_path, encoding="utf-8") as f:
        raw = json.load(f)
    results: List[Dict] = raw["results"] if isinstance(raw, dict) else raw

    # Erfolgreiche vs. fehlgeschlagene Extraktionen
    valid_results = [r for r in results if "_error" not in r]
    n_gt = len(ground_truth)
    n_results = len(results)
    n_valid = len(valid_results)

    console.print(f"\n[bold cyan]Evaluation: {config.name}[/bold cyan]")
    console.print(f"Ground Truth Items: {n_gt}")
    console.print(f"Extrahierte Items:  {n_results} ({n_valid} erfolgreich)")

    # Matching nach id (GT hat eigenes "id"-Feld, Results haben "_id")
    gt_by_id = {gt["id"]: gt for gt in ground_truth}
    results_by_id = {r["_id"]: r for r in valid_results if "_id" in r}
    common_ids = set(gt_by_id.keys()) & set(results_by_id.keys())

    # Metriken pro Feld
    field_stats = {
        f.name: {"tp": 0, "fp": 0, "fn": 0, "none_pred": 0}
        for f in config.fields
    }

    for item_id in common_ids:
        gt = gt_by_id[item_id]
        pred = results_by_id[item_id]

        for field in config.fields:
            expected = gt.get(field.name)
            predicted = pred.get(field.name)

            if predicted is None:
                field_stats[field.name]["none_pred"] += 1
                if expected is not None:
                    field_stats[field.name]["fn"] += 1
            elif is_match(field, predicted, expected):
                field_stats[field.name]["tp"] += 1
            else:
                field_stats[field.name]["fp"] += 1
                if expected is not None:
                    field_stats[field.name]["fn"] += 1

    # Ergebnistabelle
    n = len(common_ids)
    table = Table(title=f"\nEvaluationsergebnisse: {config.name}", box=box.ROUNDED)
    table.add_column("Feld", style="cyan", width=18)
    table.add_column("Precision", justify="right", width=10)
    table.add_column("Recall", justify="right", width=10)
    table.add_column("F1-Score", justify="right", width=10)
    table.add_column("None-Rate", justify="right", width=10)
    table.add_column("Korrekt/N", justify="right", width=12)

    total_tp = total_fp = total_fn = 0
    field_results = {}

    for field in config.fields:
        stats = field_stats[field.name]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        none_rate = stats["none_pred"] / n if n else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        total_tp += tp
        total_fp += fp
        total_fn += fn

        color = "green" if f1 >= 0.8 else ("yellow" if f1 >= 0.5 else "red")
        table.add_row(
            field.name,
            f"{precision:.2f}",
            f"{recall:.2f}",
            f"[{color}]{f1:.2f}[/{color}]",
            f"{none_rate:.0%}",
            f"{tp}/{n}"
        )
        field_results[field.name] = {
            "precision": precision, "recall": recall, "f1": f1, "none_rate": none_rate
        }

    # Gesamt-Zeile
    total_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * total_p * total_r / (total_p + total_r) if (total_p + total_r) > 0 else 0
    table.add_row("─" * 18, "─" * 10, "─" * 10, "─" * 10, "─" * 10, "─" * 12)
    color = "green" if total_f1 >= 0.8 else ("yellow" if total_f1 >= 0.5 else "red")
    table.add_row(
        "[bold]GESAMT[/bold]",
        f"[bold]{total_p:.2f}[/bold]",
        f"[bold]{total_r:.2f}[/bold]",
        f"[bold][{color}]{total_f1:.2f}[/{color}][/bold]",
        "",
        f"[bold]{total_tp}/{n * len(config.fields)}[/bold]"
    )

    console.print(table)

    failure_rate = (n_results - n_valid) / n_results if n_results else 0
    coverage = n / n_gt if n_gt else 0
    console.print(
        f"\n[bold]Coverage:[/bold] {coverage:.0%} ({n}/{n_gt} Items evaluiert)"
    )
    if failure_rate > 0:
        console.print(
            f"[bold yellow]Extraktionsfehler:[/bold yellow] "
            f"{failure_rate:.0%} ({n_results - n_valid}/{n_results})"
        )

    return {
        "site": config.name,
        "felder": field_results,
        "gesamt": {"precision": total_p, "recall": total_r, "f1": total_f1},
        "coverage": coverage,
        "n_items": n,
        "n_gt": n_gt,
        "extraction_failure_rate": failure_rate,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        console.print(
            "[red]Verwendung: python evaluation/evaluate_site.py <site> <results.json>[/red]"
        )
        console.print("Sites: quotes, books")
        sys.exit(1)

    site_arg = sys.argv[1]
    results_path_arg = sys.argv[2]

    if site_arg == "quotes":
        from sites.quotes_toscrape import QUOTES_CONFIG
        evaluate_site(QUOTES_CONFIG, results_path_arg)
    elif site_arg == "books":
        from sites.books_toscrape import BOOKS_LIST_CONFIG
        evaluate_site(BOOKS_LIST_CONFIG, results_path_arg)
    else:
        console.print(f"[red]Unbekannte Site: {site_arg}. Optionen: quotes, books[/red]")
        sys.exit(1)
