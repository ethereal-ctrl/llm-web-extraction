#!/usr/bin/env python3
"""
Vergleich lokales LLM vs. Cloud-LLM (OpenAI) auf identischer Datenbasis.

Motivation: Die Arbeit evaluiert ausschliesslich lokal ausgefuehrte Modelle
(qwen2.5:3b, qwen3:8b, gemma3:4b). Dieses Skript stellt denselben Lauf mit
einem Cloud-Modell daneben, um die Groessenordnung des Unterschieds zu
quantifizieren.

Methodisch identisch zur restlichen Evaluation:
  - Datenbasis  : robustness/mutations/M0/ (unmutiertes IMDb-HTML, Rang 1-50)
                  -- dieselbe Stichprobe wie Robustheitsexperiment, Ablation
                  und Kappa-Studie.
  - Cleaner     : HTMLCleaner.clean_for_imdb() (unveraendert)
  - Prompts     : PromptTemplates (System-Prompt V3 + Few-Shot), unveraendert
  - JSON-Modus  : response_format=json_object (Gegenstueck zu format="json")
  - Validierung : identische Genre-Whitelist / Laufzeit-Regex-Logik
  - Match-Logik : wortgleich zu evaluate_250.py / ablation_features.py
  - Statistik   : 95%-Bootstrap-KI (10.000 Resamples, Seed 42),
                  gepaartes Bootstrap-KI der Differenz, exakter McNemar-Test

Die lokale Baseline wird aus einem vorhandenen Output-JSON gelesen
(default: output/qwen25_250.json), auf die Schnittmenge der Snapshot-IDs
eingeschraenkt. So entsteht kein zusaetzlicher lokaler Lauf.

Verwendung:
    python evaluation/compare_cloud_model.py --model gpt-4o-mini --n 50
    python evaluation/compare_cloud_model.py --model gpt-4o --n 50
    python evaluation/compare_cloud_model.py --model gpt-4o-mini --n 3   # Testlauf

Output: evaluation/cloud_comparison_<model>.json
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from scraper.cleaner import HTMLCleaner
from llm.openai_client import OpenAIClient
from llm.prompts import PromptTemplates

GT_PATH = Path(__file__).parent / "ground_truth_250.json"
SNAP_DIR = Path(__file__).parent.parent / "robustness" / "mutations" / "M0"
FALLBACK_SNAP_DIR = Path(__file__).parent.parent / "robustness" / "snapshots"
LOCAL_BASELINE = Path(__file__).parent.parent / "output" / "qwen25_250.json"

FIELDS = ["titel", "jahr", "bewertung", "dauer_minuten", "regisseur", "genres"]
BOOTSTRAP_N = 10000
SEED = 42


# --- Match-Logik (identisch zu evaluate_250.py / ablation_features.py) ---

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


def mcnemar_exact(a_matches, b_matches):
    """
    Exakter McNemar-Test (Binomialtest auf den diskordanten Paaren).

    b01: A falsch, B richtig | b10: A richtig, B falsch.
    Zweiseitiger p-Wert ohne scipy-Abhaengigkeit.
    """
    b01 = sum(1 for a, b in zip(a_matches, b_matches) if a == 0 and b == 1)
    b10 = sum(1 for a, b in zip(a_matches, b_matches) if a == 1 and b == 0)
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "n_diskordant": 0, "p_value": 1.0}

    k = min(b01, b10)
    # P(X <= k) + P(X >= n-k) bei p=0.5, zweiseitig
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    return {"b01": b01, "b10": b10, "n_diskordant": n, "p_value": round(p, 6)}


# --- Lokale Baseline aus vorhandenem Output lesen ---

def load_local_baseline(path, wanted_ids):
    """Laedt einen bestehenden lokalen Lauf und filtert auf die Snapshot-IDs."""
    if not path.exists():
        return None, None
    data = json.loads(path.read_text(encoding="utf-8"))
    movies = data.get("movies", data if isinstance(data, list) else [])
    by_id = {m["imdb_id"]: m for m in movies if m.get("imdb_id")}
    model = (data.get("metadata") or {}).get("llm_model", path.stem)
    present = {i: by_id[i] for i in wanted_ids if i in by_id}
    return present, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI-Modell (gpt-4o-mini, gpt-4o)")
    parser.add_argument("--n", type=int, default=50,
                        help="Anzahl Filme (default 50 = Rang 1-50)")
    parser.add_argument("--baseline", default=str(LOCAL_BASELINE),
                        help="Lokaler Vergleichslauf (JSON)")
    parser.add_argument("--no-few-shot", action="store_true",
                        help="EXTRACTION_PROMPT statt Few-Shot verwenden")
    args = parser.parse_args()

    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    gt_by_id = {x["imdb_id"]: x for x in gt["filme"]}

    snap_dir = SNAP_DIR if SNAP_DIR.exists() else FALLBACK_SNAP_DIR
    snapshots = sorted(snap_dir.glob("*.html"))[:args.n]
    if not snapshots:
        print(f"Keine Snapshots gefunden in {snap_dir}")
        sys.exit(1)

    print(f"Datenbasis : {snap_dir}")
    print(f"Snapshots  : {len(snapshots)}")
    print(f"Cloud-LLM  : {args.model}\n")

    cleaner = HTMLCleaner()
    client = OpenAIClient(model=args.model)

    cloud_matches = {f: [] for f in FIELDS}
    per_film = []
    used_ids = []
    parse_failures = 0
    t0 = time.time()

    for i, snap in enumerate(snapshots, 1):
        imdb_id = snap.stem
        if imdb_id not in gt_by_id:
            continue

        html = snap.read_text(encoding="utf-8")
        cleaned = cleaner.clean_for_imdb(html)

        try:
            data = client.extract_film_data(
                cleaned, use_few_shot=not args.no_few_shot
            )
        except Exception as e:
            print(f"  [{i}] {imdb_id} FEHLER: {e}")
            data = None

        if not data:
            parse_failures += 1
            data = {}

        used_ids.append(imdb_id)
        row = {"imdb_id": imdb_id, "titel_gt": gt_by_id[imdb_id].get("titel")}
        for f in FIELDS:
            ok = field_match(f, data.get(f), gt_by_id[imdb_id].get(f))
            cloud_matches[f].append(1 if ok else 0)
            row[f] = {"pred": data.get(f), "gt": gt_by_id[imdb_id].get(f),
                      "match": bool(ok)}
        per_film.append(row)

        titel = (data.get("titel") or "?")[:40]
        treffer = sum(row[f]["match"] for f in FIELDS)
        print(f"  [{i}/{len(snapshots)}] {imdb_id} {titel:<40} {treffer}/{len(FIELDS)}")

    elapsed = time.time() - t0
    n_films = len(used_ids)
    if n_films == 0:
        print("Keine Filme mit Ground-Truth-Eintrag gefunden.")
        sys.exit(1)

    cloud_all = [m for f in FIELDS for m in cloud_matches[f]]
    cloud_f1 = {f: float(np.mean(cloud_matches[f])) for f in FIELDS}
    cloud_f1["GESAMT"] = float(np.mean(cloud_all))
    cloud_ci = {f: bootstrap_ci(cloud_matches[f]) for f in FIELDS}
    cloud_ci["GESAMT"] = bootstrap_ci(cloud_all)

    result = {
        "meta": {
            "cloud_model": args.model,
            "n_films": n_films,
            "datenbasis": str(snap_dir.relative_to(Path(__file__).parent.parent)),
            "ground_truth": "evaluation/ground_truth_250.json (Schnittmenge)",
            "prompt": "EXTRACTION_PROMPT" if args.no_few_shot else "FEW_SHOT_PROMPT",
            "json_mode": "response_format=json_object",
            "temperature": client.temperature,
            "bootstrap_n": BOOTSTRAP_N,
            "seed": SEED,
            "laufzeit_sekunden": round(elapsed, 1),
            "sekunden_pro_film": round(elapsed / n_films, 2),
            "token_usage": client.usage,
            "geschaetzte_kosten_usd": (
                round(client.estimated_cost_usd(), 4)
                if client.estimated_cost_usd() is not None else None
            ),
        },
        "cloud": {
            "f1": {k: round(v, 4) for k, v in cloud_f1.items()},
            "ci": {k: {"lo": round(v[0], 4), "hi": round(v[1], 4)}
                   for k, v in cloud_ci.items()},
            "parse_failures": parse_failures,
        },
        "per_film": per_film,
    }

    print(f"\n--- {args.model} (n={n_films}) ---")
    for f in FIELDS:
        lo, hi = cloud_ci[f]
        print(f"  {f:<15} F1={cloud_f1[f]:.4f}  [{lo:.4f}; {hi:.4f}]")
    lo, hi = cloud_ci["GESAMT"]
    print(f"  {'GESAMT':<15} F1={cloud_f1['GESAMT']:.4f}  [{lo:.4f}; {hi:.4f}]")
    print(f"  Parse-Fehler: {parse_failures}/{n_films}")
    print(f"  Laufzeit: {elapsed:.1f}s ({elapsed / n_films:.2f}s/Film)")
    if result["meta"]["geschaetzte_kosten_usd"] is not None:
        print(f"  Kosten (geschaetzt): ${result['meta']['geschaetzte_kosten_usd']:.4f}")

    # --- Vergleich gegen lokale Baseline ---
    baseline_path = Path(args.baseline)
    local, local_model = load_local_baseline(baseline_path, used_ids)

    if local:
        local_matches = {f: [] for f in FIELDS}
        common_ids = [i for i in used_ids if i in local]
        for imdb_id in common_ids:
            for f in FIELDS:
                ok = field_match(f, local[imdb_id].get(f), gt_by_id[imdb_id].get(f))
                local_matches[f].append(1 if ok else 0)

        # Cloud auf dieselbe Schnittmenge einschraenken
        idx = [used_ids.index(i) for i in common_ids]
        cloud_sub = {f: [cloud_matches[f][j] for j in idx] for f in FIELDS}

        local_all = [m for f in FIELDS for m in local_matches[f]]
        cloud_all_sub = [m for f in FIELDS for m in cloud_sub[f]]

        local_f1 = {f: float(np.mean(local_matches[f])) for f in FIELDS}
        local_f1["GESAMT"] = float(np.mean(local_all))
        local_ci = {f: bootstrap_ci(local_matches[f]) for f in FIELDS}
        local_ci["GESAMT"] = bootstrap_ci(local_all)

        d, dlo, dhi = paired_bootstrap_diff_ci(cloud_all_sub, local_all)
        mcnemar = mcnemar_exact(local_all, cloud_all_sub)

        per_field_delta = {}
        for f in FIELDS:
            fd, flo, fhi = paired_bootstrap_diff_ci(cloud_sub[f], local_matches[f])
            per_field_delta[f] = {
                "cloud": round(float(np.mean(cloud_sub[f])), 4),
                "lokal": round(float(np.mean(local_matches[f])), 4),
                "delta": round(fd, 4),
                "ci_lo": round(flo, 4),
                "ci_hi": round(fhi, 4),
            }

        result["lokal"] = {
            "model": local_model,
            "quelle": str(baseline_path),
            "n_films": len(common_ids),
            "f1": {k: round(v, 4) for k, v in local_f1.items()},
            "ci": {k: {"lo": round(v[0], 4), "hi": round(v[1], 4)}
                   for k, v in local_ci.items()},
        }
        result["vergleich"] = {
            "n_films": len(common_ids),
            "delta_gesamt_cloud_minus_lokal": round(d, 4),
            "ci_lo": round(dlo, 4),
            "ci_hi": round(dhi, 4),
            "mcnemar": mcnemar,
            "per_field": per_field_delta,
        }

        print(f"\n--- Vergleich (n={len(common_ids)} gemeinsame Filme) ---")
        print(f"  {'Feld':<15} {'lokal':>8} {'cloud':>8} {'delta':>9}")
        for f in FIELDS:
            pf = per_field_delta[f]
            print(f"  {f:<15} {pf['lokal']:>8.4f} {pf['cloud']:>8.4f} "
                  f"{pf['delta']:>+9.4f}")
        print(f"  {'GESAMT':<15} {local_f1['GESAMT']:>8.4f} "
              f"{float(np.mean(cloud_all_sub)):>8.4f} {d:>+9.4f}")
        print(f"\n  lokal = {local_model}, cloud = {args.model}")
        print(f"  Delta-F1 (cloud - lokal): {d:+.4f}  95%-KI [{dlo:+.4f}; {dhi:+.4f}]")
        print(f"  McNemar exakt: b01={mcnemar['b01']} b10={mcnemar['b10']} "
              f"p={mcnemar['p_value']}")
        if dlo <= 0 <= dhi:
            print("  -> KI enthaelt 0: kein statistisch gesicherter Unterschied.")
        else:
            besser = "Cloud" if d > 0 else "lokal"
            print(f"  -> KI schliesst 0 aus: {besser} ist gesichert besser.")
    else:
        print(f"\n(Keine lokale Baseline unter {baseline_path} gefunden - "
              f"nur Cloud-Ergebnisse berechnet.)")

    # Dateiname enthaelt Cloud-Modell UND lokale Baseline, damit sich Laeufe
    # mit unterschiedlicher Baseline nicht gegenseitig ueberschreiben.
    baseline_tag = ""
    if local and local_model:
        baseline_tag = "_vs_" + re.sub(r"[^A-Za-z0-9]+", "", local_model)
    out_path = (Path(__file__).parent /
                f"cloud_comparison_{args.model.replace('.', '')}"
                f"{baseline_tag}.json")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nGespeichert: {out_path}")


if __name__ == "__main__":
    main()
