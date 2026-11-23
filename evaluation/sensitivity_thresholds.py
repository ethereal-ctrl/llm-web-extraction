#!/usr/bin/env python3
"""
Sensitivitätsanalyse der Matching-Schwellwerte aus Tabelle 5 der Arbeit.

Variiert zwei Schwellwerte auf demselben Scraper-Output (qwen2.5:3b auf
IMDb Top 250) und protokolliert den Effekt auf den Gesamt-F1 sowie die
betroffenen feldweisen F1-Werte.

  - Bewertungstoleranz t_bew ∈ {0,05; 0,10; 0,15; 0,20; 0,25; 0,30}
  - Genre-Schwellwert  t_gen ∈ {0,3; 0,4; 0,5; 0,6; 0,7; 0,8}

Default-Eingabe: output/qwen25_250.json. Ausgabe: evaluation/sensitivity_results.json.
"""
import json
import re
import sys
from pathlib import Path
from itertools import product

import numpy as np

GT_PATH = Path(__file__).parent / "ground_truth_250.json"
DEFAULT_INPUT = Path(__file__).parent.parent / "output" / "qwen25_250.json"
OUT_PATH = Path(__file__).parent / "sensitivity_results.json"

FIELDS = ["titel", "jahr", "bewertung", "dauer_minuten", "regisseur", "genres"]

BEWERTUNG_TOL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
GENRE_THRESH = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
BOOTSTRAP_N = 10000
SEED = 42


def normalize_string(s):
    if not s:
        return ""
    return re.sub(r'\s+', ' ', str(s).strip().lower())


def field_match(field, pred, exp, t_bew, t_gen):
    if pred is None or exp is None:
        return pred == exp
    if field in ("titel", "regisseur"):
        p, e = normalize_string(pred), normalize_string(exp)
        return e in p or p in e
    if field in ("jahr", "dauer_minuten"):
        try:
            return int(pred) == int(exp)
        except Exception:
            return False
    if field == "bewertung":
        try:
            return abs(float(pred) - float(exp)) <= t_bew
        except Exception:
            return False
    if field == "genres":
        ps = {normalize_string(g) for g in (pred or [])}
        es = {normalize_string(g) for g in (exp or [])}
        if not es:
            return True
        overlap = ps & es
        prec = len(overlap) / len(ps) if ps else 0
        rec = len(overlap) / len(es) if es else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return f1 >= t_gen
    return False


def evaluate(scraper, gt_by_id, t_bew, t_gen):
    """Pro Film/Feld 0/1, anschließend feldweiser F1 + Gesamt-F1 + CI."""
    matches = {f: [] for f in FIELDS}
    for imdb_id, gt in gt_by_id.items():
        pred = scraper.get(imdb_id)
        if pred is None:
            continue
        for f in FIELDS:
            matches[f].append(
                1 if field_match(f, pred.get(f), gt.get(f), t_bew, t_gen) else 0
            )
    out = {}
    rng = np.random.default_rng(SEED)
    for f in FIELDS:
        v = matches[f]
        if not v:
            out[f] = {"f1": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n": 0}
            continue
        f1 = float(np.mean(v))
        boots = [np.mean(rng.choice(v, size=len(v), replace=True)) for _ in range(BOOTSTRAP_N)]
        out[f] = {
            "f1": round(f1, 4),
            "ci_lo": round(float(np.percentile(boots, 2.5)), 4),
            "ci_hi": round(float(np.percentile(boots, 97.5)), 4),
            "n": len(v),
        }
    all_v = [m for f in FIELDS for m in matches[f]]
    rng = np.random.default_rng(SEED)
    boots = [np.mean(rng.choice(all_v, size=len(all_v), replace=True)) for _ in range(BOOTSTRAP_N)]
    out["GESAMT"] = {
        "f1": round(float(np.mean(all_v)), 4),
        "ci_lo": round(float(np.percentile(boots, 2.5)), 4),
        "ci_hi": round(float(np.percentile(boots, 97.5)), 4),
        "n": len(all_v),
    }
    return out


def main():
    inp = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    print(f"Eingabe: {inp}")

    with open(GT_PATH) as f:
        gt_data = json.load(f)
    gt_by_id = {x["imdb_id"]: x for x in gt_data["filme"]}

    with open(inp) as f:
        scraper_raw = json.load(f)
    movies = scraper_raw.get("movies", scraper_raw if isinstance(scraper_raw, list) else [])
    scraper = {m["imdb_id"]: m for m in movies if m.get("imdb_id")}
    print(f"GT: {len(gt_by_id)}, Scraper: {len(scraper)}, "
          f"Schnittmenge: {len(set(gt_by_id) & set(scraper))}")

    results = {
        "meta": {
            "input_file": str(inp),
            "n_films": len(set(gt_by_id) & set(scraper)),
            "bewertung_tol_grid": BEWERTUNG_TOL,
            "genre_thresh_grid": GENRE_THRESH,
            "default": {"t_bew": 0.15, "t_gen": 0.5},
            "bootstrap_n": BOOTSTRAP_N,
        },
        "grid": [],
    }

    print(f"\n{'t_bew':>6} {'t_gen':>6} {'F1_bew':>8} {'F1_genres':>10} {'F1_ges':>8} "
          f"{'CI_lo':>8} {'CI_hi':>8}")
    print("-" * 70)
    for t_bew, t_gen in product(BEWERTUNG_TOL, GENRE_THRESH):
        ev = evaluate(scraper, gt_by_id, t_bew, t_gen)
        results["grid"].append({
            "t_bew": t_bew, "t_gen": t_gen,
            "fields": ev,
        })
        ges = ev["GESAMT"]
        print(f"{t_bew:>6.2f} {t_gen:>6.2f} {ev['bewertung']['f1']:>8.3f} "
              f"{ev['genres']['f1']:>10.3f} {ges['f1']:>8.3f} "
              f"{ges['ci_lo']:>8.3f} {ges['ci_hi']:>8.3f}")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nGespeichert: {OUT_PATH}")

    # Marginal-Analyse: was bewirkt nur t_bew (bei t_gen=0.5) und
    # was bewirkt nur t_gen (bei t_bew=0.15)?
    print("\n--- Marginal: t_bew bei t_gen=0.5 ---")
    print(f"{'t_bew':>6} {'F1_bew':>8} {'F1_ges':>8} {'CI':>20}")
    for entry in results["grid"]:
        if entry["t_gen"] == 0.5:
            ev = entry["fields"]
            print(f"{entry['t_bew']:>6.2f} {ev['bewertung']['f1']:>8.3f} "
                  f"{ev['GESAMT']['f1']:>8.3f}  "
                  f"[{ev['GESAMT']['ci_lo']:.3f}; {ev['GESAMT']['ci_hi']:.3f}]")

    print("\n--- Marginal: t_gen bei t_bew=0.15 ---")
    print(f"{'t_gen':>6} {'F1_genres':>10} {'F1_ges':>8} {'CI':>20}")
    for entry in results["grid"]:
        if entry["t_bew"] == 0.15:
            ev = entry["fields"]
            print(f"{entry['t_gen']:>6.2f} {ev['genres']['f1']:>10.3f} "
                  f"{ev['GESAMT']['f1']:>8.3f}  "
                  f"[{ev['GESAMT']['ci_lo']:.3f}; {ev['GESAMT']['ci_hi']:.3f}]")


if __name__ == "__main__":
    main()
