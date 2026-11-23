#!/usr/bin/env python3
"""
Evaluation für IMDb Top 250 mit Bootstrap-Konfidenzintervallen.

Verwendung:
    python evaluation/evaluate_250.py output/qwen25_250.json output/qwen3_250.json
    python evaluation/evaluate_250.py output/qwen25_250.json  # nur ein Modell
"""
import json
import sys
import re
import random
import numpy as np
from pathlib import Path
from typing import Any
from collections import defaultdict

GT_PATH = Path(__file__).parent / "ground_truth_250.json"
OUTPUT_PATH = Path(__file__).parent / "f1_250_results.json"

FELDER = {
    "titel":         "string_fuzzy",
    "jahr":          "exact",
    "bewertung":     "numeric_tolerance",
    "dauer_minuten": "exact",
    "regisseur":     "string_fuzzy",
    "genres":        "set_overlap",
}

BOOTSTRAP_N = 10000
CI_ALPHA = 0.05


def normalize_string(s: str) -> str:
    if not s:
        return ""
    return re.sub(r'\s+', ' ', s.strip().lower())


def field_match(field: str, predicted: Any, expected: Any) -> bool:
    strategy = FELDER.get(field, "exact")
    if predicted is None or expected is None:
        return predicted == expected

    if strategy == "exact":
        return predicted == expected
    elif strategy == "string_fuzzy":
        pred = normalize_string(str(predicted))
        exp = normalize_string(str(expected))
        return exp in pred or pred in exp
    elif strategy == "numeric_tolerance":
        try:
            return abs(float(predicted) - float(expected)) <= 0.15
        except (ValueError, TypeError):
            return False
    elif strategy == "set_overlap":
        pred_set = {normalize_string(g) for g in (predicted or [])}
        exp_set = {normalize_string(g) for g in (expected or [])}
        if not exp_set:
            return True
        overlap = pred_set & exp_set
        precision = len(overlap) / len(pred_set) if pred_set else 0
        recall = len(overlap) / len(exp_set) if exp_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1 >= 0.5
    return predicted == expected


def genre_f1(predicted, expected):
    """Compute per-item genre F1 (micro) for a single film."""
    pred_set = {normalize_string(g) for g in (predicted or [])}
    exp_set = {normalize_string(g) for g in (expected or [])}
    if not exp_set and not pred_set:
        return 1.0
    if not exp_set:
        return 1.0
    if not pred_set:
        return 0.0
    overlap = pred_set & exp_set
    p = len(overlap) / len(pred_set)
    r = len(overlap) / len(exp_set)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate_model(scraper_path: str, ground_truth: dict) -> dict:
    """Evaluate a single model's output against ground truth."""
    with open(scraper_path) as f:
        scraper_data = json.load(f)

    scraper_films = {}
    movies = scraper_data.get("movies", scraper_data if isinstance(scraper_data, list) else [])
    for m in movies:
        if m.get("imdb_id"):
            scraper_films[m["imdb_id"]] = m

    common_ids = sorted(set(ground_truth.keys()) & set(scraper_films.keys()))
    n = len(common_ids)

    print(f"\n{'='*60}")
    print(f"Evaluation: {Path(scraper_path).name}")
    print(f"GT: {len(ground_truth)} | Scraper: {len(scraper_films)} | Gemeinsam: {n}")
    print(f"Coverage: {n/len(ground_truth):.1%}")

    # Per-item binary match vectors
    match_vectors = {field: [] for field in FELDER}
    # For genres: also compute micro P/R/F1
    genre_p_list, genre_r_list = [], []

    errors = []  # Track individual errors for classification

    for imdb_id in common_ids:
        gt = ground_truth[imdb_id]
        pred = scraper_films[imdb_id]

        for field in FELDER:
            expected = gt.get(field)
            predicted = pred.get(field)

            if field == "genres":
                pred_set = {normalize_string(g) for g in (predicted or [])}
                exp_set = {normalize_string(g) for g in (expected or [])}
                if exp_set:
                    overlap = pred_set & exp_set
                    p = len(overlap) / len(pred_set) if pred_set else 0
                    r = len(overlap) / len(exp_set) if exp_set else 0
                    genre_p_list.append(p)
                    genre_r_list.append(r)
                matched = field_match(field, predicted, expected)
            else:
                matched = field_match(field, predicted, expected)

            match_vectors[field].append(1 if matched else 0)

            if not matched:
                errors.append({
                    "imdb_id": imdb_id,
                    "field": field,
                    "expected": expected,
                    "predicted": predicted,
                    "titel": gt.get("titel", "?"),
                })

    # Compute per-field F1 (binary match rate = P = R = F1 for exact fields)
    results = {}
    print(f"\n{'Feld':<18} {'P':>8} {'R':>8} {'F1':>8} {'CI_lo':>8} {'CI_hi':>8} {'OK/N':>8}")
    print("-" * 70)

    total_tp = 0
    total_n = 0

    for field in FELDER:
        vec = match_vectors[field]
        tp = sum(vec)
        total_tp += tp
        total_n += len(vec)

        if field == "genres" and genre_p_list:
            # Micro-averaged genre P/R/F1
            p = np.mean(genre_p_list)
            r = np.mean(genre_r_list)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

            # Bootstrap CI for genre F1
            rng = np.random.default_rng(42)
            f1_boots = []
            for _ in range(BOOTSTRAP_N):
                idx = rng.integers(0, len(genre_p_list), len(genre_p_list))
                bp = np.mean([genre_p_list[i] for i in idx])
                br = np.mean([genre_r_list[i] for i in idx])
                bf1 = 2 * bp * br / (bp + br) if (bp + br) > 0 else 0
                f1_boots.append(bf1)
            ci_lo = np.percentile(f1_boots, 100 * CI_ALPHA / 2)
            ci_hi = np.percentile(f1_boots, 100 * (1 - CI_ALPHA / 2))
        else:
            p = r = f1 = tp / len(vec) if vec else 0
            # Bootstrap CI
            rng = np.random.default_rng(42)
            f1_boots = []
            for _ in range(BOOTSTRAP_N):
                idx = rng.integers(0, len(vec), len(vec))
                bf1 = np.mean([vec[i] for i in idx])
                f1_boots.append(bf1)
            ci_lo = np.percentile(f1_boots, 100 * CI_ALPHA / 2)
            ci_hi = np.percentile(f1_boots, 100 * (1 - CI_ALPHA / 2))

        results[field] = {
            "p": round(p, 4), "r": round(r, 4), "f1": round(f1, 4),
            "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
            "n": len(vec), "tp": tp,
        }
        print(f"{field:<18} {p:>8.3f} {r:>8.3f} {f1:>8.3f} {ci_lo:>8.3f} {ci_hi:>8.3f} {tp:>4}/{len(vec)}")

    # Gesamt (micro)
    total_p = total_r = total_f1 = total_tp / total_n if total_n > 0 else 0
    # Bootstrap CI for total
    all_matches = []
    for field in FELDER:
        all_matches.extend(match_vectors[field])
    rng = np.random.default_rng(42)
    total_boots = []
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, len(all_matches), len(all_matches))
        total_boots.append(np.mean([all_matches[i] for i in idx]))
    total_ci_lo = np.percentile(total_boots, 100 * CI_ALPHA / 2)
    total_ci_hi = np.percentile(total_boots, 100 * (1 - CI_ALPHA / 2))

    results["GESAMT"] = {
        "p": round(total_p, 4), "r": round(total_r, 4), "f1": round(total_f1, 4),
        "ci_lo": round(total_ci_lo, 4), "ci_hi": round(total_ci_hi, 4),
        "n": total_n, "tp": total_tp,
    }
    print("-" * 70)
    print(f"{'GESAMT':<18} {total_p:>8.3f} {total_r:>8.3f} {total_f1:>8.3f} {total_ci_lo:>8.3f} {total_ci_hi:>8.3f} {total_tp:>4}/{total_n}")

    # Error classification
    print(f"\n--- Fehlerübersicht: {len(errors)} Fehler aus {total_n} Vergleichen ---")
    error_by_field = defaultdict(int)
    for e in errors:
        error_by_field[e["field"]] += 1
    for field, count in sorted(error_by_field.items(), key=lambda x: -x[1]):
        print(f"  {field}: {count} Fehler")

    # Print first 10 errors for inspection
    print(f"\nErste 10 Fehler:")
    for e in errors[:10]:
        print(f"  [{e['field']}] {e['titel']}: erwartet={e['expected']}, extrahiert={e['predicted']}")

    return {
        "results": results,
        "coverage": n / len(ground_truth),
        "n_common": n,
        "n_gt": len(ground_truth),
        "n_scraper": len(scraper_films),
        "errors": errors,
        "model_file": Path(scraper_path).name,
    }


def compute_timing(scraper_path: str) -> dict:
    """Extract timing info from scraper metadata if available."""
    with open(scraper_path) as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    return {
        "total_time_s": meta.get("total_time_seconds"),
        "avg_time_per_film_s": meta.get("avg_time_per_film"),
        "model": meta.get("model"),
        "n_films": meta.get("total_processed"),
    }


def main():
    if len(sys.argv) < 2:
        print("Verwendung: python evaluate_250.py <scraper_output.json> [<scraper_output_2.json>]")
        sys.exit(1)

    # Load GT
    with open(GT_PATH) as f:
        gt_data = json.load(f)
    ground_truth = {film["imdb_id"]: film for film in gt_data["filme"]}
    print(f"Ground Truth: {len(ground_truth)} Filme")

    all_results = {}
    for path in sys.argv[1:]:
        model_eval = evaluate_model(path, ground_truth)
        timing = compute_timing(path)
        model_eval["timing"] = timing
        label = Path(path).stem
        all_results[label] = model_eval

    # Save results
    # Strip non-serializable errors for JSON output
    save_data = {}
    for label, data in all_results.items():
        save_data[label] = {
            "results": data["results"],
            "coverage": data["coverage"],
            "n_common": data["n_common"],
            "timing": data["timing"],
            "model_file": data["model_file"],
        }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nErgebnisse gespeichert: {OUTPUT_PATH}")

    # If two models, print comparison
    if len(all_results) == 2:
        labels = list(all_results.keys())
        print(f"\n{'='*60}")
        print(f"MODELLVERGLEICH: {labels[0]} vs {labels[1]}")
        print(f"{'='*60}")
        print(f"{'Feld':<18} {'F1 '+labels[0][:10]:>12} {'F1 '+labels[1][:10]:>12} {'Diff':>8}")
        print("-" * 52)
        for field in list(FELDER.keys()) + ["GESAMT"]:
            f1_a = all_results[labels[0]]["results"][field]["f1"]
            f1_b = all_results[labels[1]]["results"][field]["f1"]
            diff = f1_b - f1_a
            sign = "+" if diff > 0 else ""
            print(f"{field:<18} {f1_a:>12.3f} {f1_b:>12.3f} {sign}{diff:>7.3f}")


if __name__ == "__main__":
    main()
