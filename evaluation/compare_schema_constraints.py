#!/usr/bin/env python3
"""
A/B-Vergleich: sechsstufige Fallback-Kaskade vs. JSON-Schema-Constraints.

Aufgabe aus dem Ausblick (Abschnitt 6.7) der Bachelorarbeit. Beide
Pfade laufen mit identischem Modell, identischem Prompt, identischer
HTML-Bereinigung — der einzige Unterschied ist:

  Pfad A:  format="json"   (Syntax-Garantie, Schema per Prompt)
           Parse über sechsstufige Fallback-Kaskade
           (entspricht der Originalpipeline)

  Pfad B:  format=<JSON-Schema>  (Decoder-Constraint via Ollama 0.5+)
           Direkter json.loads, da das Schema bereits Syntax und
           Feldtypen garantiert.

Auf den vorhandenen 50 IMDb-Snapshots (robustness/snapshots/M0)
werden beide Pfade ausgeführt, F1, Parse-Fehler, Latenz und
Coverage protokolliert.

Output: evaluation/schema_constraints_results.json
"""
import sys
import json
import re
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scraper.cleaner import HTMLCleaner
from llm.ollama_client import OllamaClient

GT_PATH = Path(__file__).parent / "ground_truth_250.json"
SNAP_DIR = Path(__file__).parent.parent / "robustness" / "snapshots"
OUT_PATH = Path(__file__).parent / "schema_constraints_results.json"

FIELDS = ["titel", "jahr", "bewertung", "dauer_minuten", "regisseur", "genres"]
BOOTSTRAP_N = 10000
SEED = 42


def normalize_string(s):
    if not s:
        return ""
    return re.sub(r'\s+', ' ', str(s).strip().lower())


def field_match(field, pred, exp):
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
            return abs(float(pred) - float(exp)) <= 0.15
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
        return f1 >= 0.5
    return False


def bootstrap_ci(values, n_boot=BOOTSTRAP_N, seed=SEED):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0)
    boots = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    return (
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )


def paired_diff_ci(a, b, n_boot=BOOTSTRAP_N, seed=SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    if n == 0:
        return (0.0, 0.0, 0.0)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(np.mean(a[idx]) - np.mean(b[idx]))
    return (float(np.mean(a) - np.mean(b)),
            float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)))


def run_pipeline(client, cleaner, html, mode):
    cleaned = cleaner.clean_for_imdb(html)
    if len(cleaned) < 100:
        cleaned = cleaner.clean(html)
    t0 = time.time()
    if mode == "fallback":
        data = client.extract_film_data(cleaned, use_few_shot=True)
    elif mode == "schema":
        data = client.extract_film_data_schema(cleaned, use_few_shot=True)
    else:
        raise ValueError(mode)
    elapsed = time.time() - t0
    return data, elapsed


def evaluate_set(predictions, gt_by_id):
    """predictions: dict imdb_id -> {fields..., success: bool, time_s: float}."""
    matches = {f: [] for f in FIELDS}
    parse_failures = 0
    latencies = []
    for imdb_id, pred in predictions.items():
        if imdb_id not in gt_by_id:
            continue
        if not pred.get("success"):
            parse_failures += 1
            for f in FIELDS:
                matches[f].append(0)
        else:
            for f in FIELDS:
                matches[f].append(
                    1 if field_match(f, pred.get(f), gt_by_id[imdb_id].get(f)) else 0
                )
        if pred.get("time_s") is not None:
            latencies.append(pred["time_s"])
    n = len(predictions)
    f1 = {f: float(np.mean(matches[f])) if matches[f] else 0 for f in FIELDS}
    f1_total = float(np.mean([m for f in FIELDS for m in matches[f]])) if n else 0
    return {
        "n": n,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / n if n else 0,
        "f1": {**{k: round(v, 4) for k, v in f1.items()}, "GESAMT": round(f1_total, 4)},
        "matches": matches,
        "avg_latency_s": float(np.mean(latencies)) if latencies else 0,
        "median_latency_s": float(np.median(latencies)) if latencies else 0,
    }


def main():
    with open(GT_PATH) as f:
        gt_data = json.load(f)
    gt_by_id = {x["imdb_id"]: x for x in gt_data["filme"]}

    snapshots = sorted(SNAP_DIR.glob("*.html"))
    print(f"Snapshots: {len(snapshots)}")

    cleaner = HTMLCleaner()
    client = OllamaClient(model="qwen2.5:3b")
    if not client.is_available():
        print("Ollama nicht erreichbar.")
        sys.exit(1)

    pred_fallback = {}
    pred_schema = {}

    print("\n=== Pfad A: Fallback-Kaskade (format='json') ===")
    for i, snap in enumerate(snapshots, 1):
        imdb_id = snap.stem
        if imdb_id not in gt_by_id:
            continue
        html = snap.read_text(encoding="utf-8")
        data, t = run_pipeline(client, cleaner, html, "fallback")
        ok = data is not None
        rec = {"success": ok, "time_s": t}
        if ok:
            for f in FIELDS:
                rec[f] = data.get(f)
        pred_fallback[imdb_id] = rec
        marker = "OK" if ok else "FAIL"
        print(f"  [{i:>2}/{len(snapshots)}] {imdb_id} {marker} {t:.2f}s")

    print("\n=== Pfad B: JSON-Schema-Constraints (format=<schema>) ===")
    for i, snap in enumerate(snapshots, 1):
        imdb_id = snap.stem
        if imdb_id not in gt_by_id:
            continue
        html = snap.read_text(encoding="utf-8")
        data, t = run_pipeline(client, cleaner, html, "schema")
        ok = data is not None
        rec = {"success": ok, "time_s": t}
        if ok:
            for f in FIELDS:
                rec[f] = data.get(f)
        pred_schema[imdb_id] = rec
        marker = "OK" if ok else "FAIL"
        print(f"  [{i:>2}/{len(snapshots)}] {imdb_id} {marker} {t:.2f}s")

    res_fb = evaluate_set(pred_fallback, gt_by_id)
    res_sc = evaluate_set(pred_schema, gt_by_id)

    # Bootstrap-CIs für Gesamt-F1 und feldweise
    fb_ci = {}
    sc_ci = {}
    delta = {}
    for f in FIELDS:
        a = res_fb["matches"][f]
        b = res_sc["matches"][f]
        fb_ci[f] = bootstrap_ci(a)
        sc_ci[f] = bootstrap_ci(b)
        d, lo, hi = paired_diff_ci(b, a)  # Schema − Fallback
        delta[f] = {"delta": round(d, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}

    a_all = [m for f in FIELDS for m in res_fb["matches"][f]]
    b_all = [m for f in FIELDS for m in res_sc["matches"][f]]
    fb_ci["GESAMT"] = bootstrap_ci(a_all)
    sc_ci["GESAMT"] = bootstrap_ci(b_all)
    d, lo, hi = paired_diff_ci(b_all, a_all)
    delta["GESAMT"] = {"delta": round(d, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}

    out = {
        "meta": {
            "n_films": res_fb["n"],
            "model": "qwen2.5:3b",
            "ollama_version": "0.16.2",
            "bootstrap_n": BOOTSTRAP_N,
            "snapshots": str(SNAP_DIR),
            "description": "JSON-Schema-Constraints vs. Fallback-Kaskade",
        },
        "fallback": {
            **{k: v for k, v in res_fb.items() if k != "matches"},
            "ci": {k: {"lo": round(v[0], 4), "hi": round(v[1], 4)} for k, v in fb_ci.items()},
        },
        "schema": {
            **{k: v for k, v in res_sc.items() if k != "matches"},
            "ci": {k: {"lo": round(v[0], 4), "hi": round(v[1], 4)} for k, v in sc_ci.items()},
        },
        "delta_schema_minus_fallback": delta,
    }

    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nGespeichert: {OUT_PATH}")

    print(f"\n{'Pfad':<10} {'n':>4} {'fail':>5} {'F1':>8} {'CI':>20} {'avg_t':>8} {'med_t':>8}")
    print("-" * 70)
    for label, r, ci in [("fallback", res_fb, fb_ci), ("schema", res_sc, sc_ci)]:
        print(f"{label:<10} {r['n']:>4} {r['parse_failures']:>5} "
              f"{r['f1']['GESAMT']:>8.3f}  [{ci['GESAMT'][0]:.3f};{ci['GESAMT'][1]:.3f}] "
              f"{r['avg_latency_s']:>8.2f} {r['median_latency_s']:>8.2f}")
    print(f"\nΔF1 (Schema − Fallback) = {delta['GESAMT']['delta']:+.3f}  "
          f"95%-KI [{delta['GESAMT']['ci_lo']:+.3f}; {delta['GESAMT']['ci_hi']:+.3f}]")


if __name__ == "__main__":
    main()
