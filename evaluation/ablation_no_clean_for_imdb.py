#!/usr/bin/env python3
"""
Zusatz-Ablation: F1 ohne IMDb-spezifische Vorverarbeitung clean_for_imdb().

Die bestehende Ablation (Tabelle 13 der Arbeit) deaktiviert nur drei
einzelne Optimierungs-Bausteine (Genre-Whitelist, Laufzeit-Regex,
JSON-Modus). Der eigentliche IMDb-Cleaner mit data-testid-Selektoren
wird dabei nie deaktiviert. Diese Ablation schliesst diese Lücke:

  v_full:    cleaner.clean_for_imdb(html)   <-- IMDb-bewusste Selektoren
  v_generic: cleaner.clean(html)            <-- generisch, keine IMDb-Selektoren

Beide Pfade laufen anschliessend durch dieselbe LLM-Pipeline (qwen2.5:3b,
Few-Shot, JSON-Modus, Genre-Whitelist im Postprocessing). Es bleibt also
nur die Frage: Wieviel der Extraktionsleistung steckt im IMDb-bewussten
Cleaner-Pfad, wieviel im LLM?

Datenbasis: Die 50 M0-Snapshots des Robustheitsexperiments
(robustness/mutations/M0/), unmutiertes IMDb-HTML zum Stichtag.
Ground Truth: evaluation/ground_truth_250.json, Schnittmenge mit den 50
Snapshots.

Output: evaluation/ablation_no_clean_for_imdb_results.json
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scraper.cleaner import HTMLCleaner
from llm.ollama_client import OllamaClient

GT_PATH = Path(__file__).parent / "ground_truth_250.json"
SNAP_DIR = Path(__file__).parent.parent / "robustness" / "mutations" / "M0"
OUT_PATH = Path(__file__).parent / "ablation_no_clean_for_imdb_results.json"

FIELDS = ["titel", "jahr", "bewertung", "dauer_minuten", "regisseur", "genres"]
BOOTSTRAP_N = 10000
SEED = 42


# --- match logic (identisch zu evaluate_250.py) ---

def normalize_string(s):
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())


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
    boots = [np.mean(rng.choice(arr, size=len(arr), replace=True))
             for _ in range(n_boot)]
    return (float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


def paired_bootstrap_diff_ci(a, b, n_boot=BOOTSTRAP_N, seed=SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(np.mean(a[idx]) - np.mean(b[idx]))
    diffs = np.asarray(diffs)
    return (float(np.mean(a) - np.mean(b)),
            float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)))


def evaluate_pipeline(label, cleaner_fn, snapshots, gt_by_id, client):
    matches = {f: [] for f in FIELDS}
    n = 0
    parse_failures = 0
    latencies = []
    for snap in snapshots:
        imdb_id = snap.stem
        if imdb_id not in gt_by_id:
            continue
        n += 1
        html = snap.read_text(encoding="utf-8")
        cleaned = cleaner_fn(html)
        t0 = time.time()
        try:
            data = client.extract_film_data(cleaned, use_few_shot=True)
        except Exception as e:
            print(f"  [{label}] {imdb_id} ERROR: {e}")
            data = None
        latencies.append(time.time() - t0)
        if not data:
            parse_failures += 1
            data = {f: None for f in FIELDS}
        for f in FIELDS:
            ok = field_match(f, data.get(f), gt_by_id[imdb_id].get(f))
            matches[f].append(1 if ok else 0)

    f1 = {f: float(np.mean(matches[f])) for f in FIELDS} if n else {}
    all_v = [m for f in FIELDS for m in matches[f]]
    f1["GESAMT"] = float(np.mean(all_v)) if all_v else 0.0
    cis = {f: bootstrap_ci(matches[f]) for f in FIELDS}
    cis["GESAMT"] = bootstrap_ci(all_v)
    return {
        "n": n,
        "f1": {k: round(v, 4) for k, v in f1.items()},
        "ci": {k: {"lo": round(v[0], 4), "hi": round(v[1], 4)}
               for k, v in cis.items()},
        "matches": matches,
        "parse_failures": parse_failures,
        "avg_latency_s": float(np.mean(latencies)) if latencies else 0.0,
    }


def main():
    with open(GT_PATH) as f:
        gt = json.load(f)
    gt_by_id = {x["imdb_id"]: x for x in gt["filme"]}

    snapshots = sorted(SNAP_DIR.glob("*.html"))
    print(f"Snapshots: {len(snapshots)}")

    cleaner = HTMLCleaner()
    client = OllamaClient(model="qwen2.5:3b")
    if not client.is_available():
        print("Ollama nicht erreichbar.")
        sys.exit(1)

    print("\n=== v_full: clean_for_imdb (IMDb-bewusste data-testid-Selektoren) ===")
    full = evaluate_pipeline(
        "full", lambda h: cleaner.clean_for_imdb(h),
        snapshots, gt_by_id, client)
    print(f"  F1 Gesamt: {full['f1']['GESAMT']:.4f} "
          f"[{full['ci']['GESAMT']['lo']:.4f}; {full['ci']['GESAMT']['hi']:.4f}]")
    print(f"  Parse-Fehler: {full['parse_failures']}/{full['n']}")
    print(f"  Avg Latenz: {full['avg_latency_s']:.1f}s")

    print("\n=== v_generic: clean (rein generisch, KEINE IMDb-Selektoren) ===")
    generic = evaluate_pipeline(
        "generic", lambda h: cleaner.clean(h),
        snapshots, gt_by_id, client)
    print(f"  F1 Gesamt: {generic['f1']['GESAMT']:.4f} "
          f"[{generic['ci']['GESAMT']['lo']:.4f}; {generic['ci']['GESAMT']['hi']:.4f}]")
    print(f"  Parse-Fehler: {generic['parse_failures']}/{generic['n']}")
    print(f"  Avg Latenz: {generic['avg_latency_s']:.1f}s")

    # Gepaartes Bootstrap fuer Delta_F1
    deltas = {}
    for f in FIELDS:
        a = full["matches"][f]
        b = generic["matches"][f]
        m = min(len(a), len(b))
        d, lo, hi = paired_bootstrap_diff_ci(a[:m], b[:m])
        deltas[f] = {"delta": round(d, 4),
                     "ci_lo": round(lo, 4),
                     "ci_hi": round(hi, 4)}
    a_all = [m for f in FIELDS for m in full["matches"][f]]
    b_all = [m for f in FIELDS for m in generic["matches"][f]]
    m = min(len(a_all), len(b_all))
    d, lo, hi = paired_bootstrap_diff_ci(a_all[:m], b_all[:m])
    deltas["GESAMT"] = {"delta": round(d, 4),
                        "ci_lo": round(lo, 4),
                        "ci_hi": round(hi, 4)}

    out = {
        "meta": {
            "n_films": full["n"],
            "model": "qwen2.5:3b",
            "ground_truth": "evaluation/ground_truth_250.json (Schnittmenge mit M0-Snapshots)",
            "description": (
                "Ablation: v_full nutzt clean_for_imdb() mit IMDb-bewussten "
                "data-testid-Selektoren; v_generic nutzt das rein generische "
                "clean() ohne IMDb-spezifische Vorextraktion. Beide Pfade "
                "laufen durch dieselbe LLM-Pipeline (qwen2.5:3b, Few-Shot, "
                "JSON-Modus, Genre-Whitelist-Postprocessing). Differenz "
                "Delta_F1 = F1(v_full) - F1(v_generic) zeigt den Beitrag "
                "der IMDb-bewussten Vorextraktion."
            ),
            "bootstrap_n": BOOTSTRAP_N,
        },
        "v_full": {k: full[k] for k in ("n", "f1", "ci", "parse_failures", "avg_latency_s")},
        "v_generic": {k: generic[k] for k in ("n", "f1", "ci", "parse_failures", "avg_latency_s")},
        "delta": deltas,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\n--- Delta F1 (v_full - v_generic) ---")
    print(f"{'Feld':<16} {'F_full':>8} {'F_gen':>8} {'Delta':>8} {'CI_lo':>8} {'CI_hi':>8}")
    for f in FIELDS + ["GESAMT"]:
        ff = full["f1"][f]
        fg = generic["f1"][f]
        dd = deltas[f]
        print(f"{f:<16} {ff:>8.3f} {fg:>8.3f} "
              f"{dd['delta']:>+8.3f} {dd['ci_lo']:>+8.3f} {dd['ci_hi']:>+8.3f}")
    print(f"\nGespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
