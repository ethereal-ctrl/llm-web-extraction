#!/usr/bin/env python3
"""
Echte Feature-Ablation der drei Optimierungs-Bausteine
(Genre-Whitelist, deterministischer Laufzeit-Regex, JSON-Modus).

Bislang waren die drei Beiträge in Tabelle 10 nur aus den feldweisen
F1-Werten abgeschaetzt. Dieses Skript fuehrt sie als echte Ablation
durch: Auf denselben 50 M0-Snapshots (Rang 1-50) laeuft die volle
Pipeline einmal als Referenz und dann je einmal mit genau einem
deaktivierten Baustein. Alle vier Laeufe nutzen qwen2.5:3b, Few-Shot
und dieselbe Match-Logik wie evaluate_250.py / ablation_no_clean_for_imdb.py.

Abgeschaltet wird jeweils OHNE den Produktivcode zu aendern, ueber lokale
Nachbau-Funktionen der relevanten Validierungsschritte:

  full        : volle Pipeline (Referenz)
  no_whitelist: Genres roh aus dem LLM, KEIN Whitelist-Filter
  no_regex    : dauer_minuten ohne "2h 22m"->142-Umrechnung (nur int())
  no_jsonmode : LLM-Aufruf OHNE format="json" (kein Decoder-Constraint)

Datenbasis: robustness/mutations/M0/ (unmutiertes IMDb-HTML, Rang 1-50).
Ground Truth: evaluation/ground_truth_250.json (Schnittmenge).
Output: evaluation/ablation_features_results.json
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
from llm.prompts import PromptTemplates

GT_PATH = Path(__file__).parent / "ground_truth_250.json"
SNAP_DIR = Path(__file__).parent.parent / "robustness" / "mutations" / "M0"
OUT_PATH = Path(__file__).parent / "ablation_features_results.json"

FIELDS = ["titel", "jahr", "bewertung", "dauer_minuten", "regisseur", "genres"]
BOOTSTRAP_N = 10000
SEED = 42


# --- match logic (identisch zu evaluate_250.py / ablation_no_clean_for_imdb.py) ---

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


# --- Validierungs-Nachbau mit gezielt abschaltbaren Bausteinen ---

def validate(data, *, whitelist=True, regex=True):
    """Nachbau von _validate_film_data; whitelist/regex einzeln abschaltbar.

    Die uebrigen Felder (titel/jahr/bewertung/regisseur) werden identisch
    zur Produktivlogik behandelt, damit nur der jeweils getestete Baustein
    den Unterschied macht.
    """
    v = {}
    for field in ["titel", "regisseur"]:
        val = data.get(field)
        v[field] = val.strip() if isinstance(val, str) and val else None

    # jahr
    val = data.get("jahr")
    if val is not None:
        try:
            if isinstance(val, str):
                nums = re.findall(r"\d+", val)
                v["jahr"] = int(nums[0]) if nums else None
            else:
                v["jahr"] = int(val)
        except (ValueError, TypeError):
            v["jahr"] = None
    else:
        v["jahr"] = None

    # dauer_minuten
    val = data.get("dauer_minuten")
    if val is not None:
        try:
            if isinstance(val, str):
                if regex:
                    h = re.search(r"(\d+)\s*h", val)
                    mm = re.search(r"(\d+)\s*m(?:in)?", val)
                    if h or mm:
                        v["dauer_minuten"] = (int(h.group(1)) if h else 0) * 60 + \
                                             (int(mm.group(1)) if mm else 0)
                    else:
                        nums = re.findall(r"\d+", val)
                        v["dauer_minuten"] = int(nums[0]) if nums else None
                else:
                    # OHNE Regex: nur direkte int-Konvertierung, sonst None.
                    # "2h 22m" laesst sich so nicht parsen -> None.
                    v["dauer_minuten"] = int(val)
            else:
                v["dauer_minuten"] = int(val)
        except (ValueError, TypeError):
            v["dauer_minuten"] = None
    else:
        v["dauer_minuten"] = None

    # bewertung
    val = data.get("bewertung")
    if val is not None:
        try:
            if isinstance(val, str):
                val = val.replace("/10", "").strip()
            v["bewertung"] = float(val)
        except (ValueError, TypeError):
            v["bewertung"] = None
    else:
        v["bewertung"] = None

    # genres
    genres = data.get("genres", data.get("genre", []))
    if isinstance(genres, str):
        gl = [g.strip() for g in genres.split(",")]
    elif isinstance(genres, list):
        gl = [str(g).strip() for g in genres if g]
    else:
        gl = []
    if whitelist:
        filtered = [g for g in gl if g in PromptTemplates.VALID_IMDB_GENRES]
        if not filtered and gl:
            vlow = {x.lower(): x for x in PromptTemplates.VALID_IMDB_GENRES}
            filtered = [vlow[g.lower()] for g in gl if g.lower() in vlow]
        v["genres"] = filtered
    else:
        # OHNE Whitelist: rohe Genres unveraendert uebernehmen
        v["genres"] = gl
    return v


def llm_raw(client, cleaned, *, json_mode=True):
    """Roh-Extraktion ueber den Ollama-Client, JSON-Modus abschaltbar."""
    user_prompt = PromptTemplates.get_few_shot_prompt(cleaned)
    kwargs = dict(
        model=client.model,
        messages=[
            {"role": "system", "content": PromptTemplates.get_system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        options=client.options,
    )
    if json_mode:
        kwargs["format"] = "json"
    resp = client.client.chat(**kwargs)
    raw = resp.message.content
    return client.extract_json(raw)


def evaluate(label, cleaner, client, snapshots, gt_by_id,
             *, whitelist=True, regex=True, json_mode=True):
    matches = {f: [] for f in FIELDS}
    n = 0
    parse_failures = 0
    for snap in snapshots:
        imdb_id = snap.stem
        if imdb_id not in gt_by_id:
            continue
        n += 1
        html = snap.read_text(encoding="utf-8")
        cleaned = cleaner.clean_for_imdb(html)
        try:
            data = llm_raw(client, cleaned, json_mode=json_mode)
        except Exception as e:
            print(f"  [{label}] {imdb_id} ERROR: {e}")
            data = None
        if not data:
            parse_failures += 1
            data = {}
        v = validate(data, whitelist=whitelist, regex=regex)
        for f in FIELDS:
            ok = field_match(f, v.get(f), gt_by_id[imdb_id].get(f))
            matches[f].append(1 if ok else 0)

    all_v = [m for f in FIELDS for m in matches[f]]
    f1 = {f: float(np.mean(matches[f])) for f in FIELDS}
    f1["GESAMT"] = float(np.mean(all_v)) if all_v else 0.0
    cis = {f: bootstrap_ci(matches[f]) for f in FIELDS}
    cis["GESAMT"] = bootstrap_ci(all_v)
    print(f"  [{label}] F1 Gesamt: {f1['GESAMT']:.4f} "
          f"[{cis['GESAMT'][0]:.4f}; {cis['GESAMT'][1]:.4f}]  "
          f"parse_fail={parse_failures}/{n}")
    return {
        "n": n,
        "f1": {k: round(v, 4) for k, v in f1.items()},
        "ci": {k: {"lo": round(v[0], 4), "hi": round(v[1], 4)} for k, v in cis.items()},
        "matches": matches,
        "parse_failures": parse_failures,
    }


def main():
    gt = json.load(open(GT_PATH))
    gt_by_id = {x["imdb_id"]: x for x in gt["filme"]}
    snapshots = sorted(SNAP_DIR.glob("*.html"))
    print(f"Snapshots: {len(snapshots)}")

    cleaner = HTMLCleaner()
    client = OllamaClient(model="qwen2.5:3b")
    if not client.is_available():
        print("Ollama nicht erreichbar.")
        sys.exit(1)

    print("\n=== full (Referenz: alle Bausteine aktiv) ===")
    full = evaluate("full", cleaner, client, snapshots, gt_by_id)

    print("\n=== no_whitelist (Genre-Whitelist AUS) ===")
    no_wl = evaluate("no_whitelist", cleaner, client, snapshots, gt_by_id,
                     whitelist=False)

    print("\n=== no_regex (Laufzeit-Regex AUS) ===")
    no_rx = evaluate("no_regex", cleaner, client, snapshots, gt_by_id,
                     regex=False)

    print("\n=== no_jsonmode (JSON-Modus AUS) ===")
    no_jm = evaluate("no_jsonmode", cleaner, client, snapshots, gt_by_id,
                     json_mode=False)

    variants = {"no_whitelist": no_wl, "no_regex": no_rx, "no_jsonmode": no_jm}
    deltas = {}
    for name, var in variants.items():
        a_all = [m for f in FIELDS for m in full["matches"][f]]
        b_all = [m for f in FIELDS for m in var["matches"][f]]
        mlen = min(len(a_all), len(b_all))
        d, lo, hi = paired_bootstrap_diff_ci(a_all[:mlen], b_all[:mlen])
        deltas[name] = {
            "f1_full": full["f1"]["GESAMT"],
            "f1_ohne": var["f1"]["GESAMT"],
            "delta_full_minus_ohne": round(d, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
        }

    out = {
        "meta": {
            "n_films": full["n"],
            "model": "qwen2.5:3b",
            "datenbasis": "robustness/mutations/M0/ (unmutiertes IMDb-HTML, Rang 1-50)",
            "ground_truth": "evaluation/ground_truth_250.json (Schnittmenge)",
            "bootstrap_n": BOOTSTRAP_N,
            "description": (
                "Echte Feature-Ablation: full = alle Bausteine aktiv; je Variante "
                "wird genau ein Baustein deaktiviert (Genre-Whitelist, Laufzeit-Regex "
                "'2h22m'->142, JSON-Modus format=json). delta = F1(full) - F1(ohne), "
                "gepaartes 95%-Bootstrap-KI. Cleaner bleibt clean_for_imdb()."
            ),
        },
        "full": {k: full[k] for k in ("n", "f1", "ci", "parse_failures")},
        "no_whitelist": {k: no_wl[k] for k in ("n", "f1", "ci", "parse_failures")},
        "no_regex": {k: no_rx[k] for k in ("n", "f1", "ci", "parse_failures")},
        "no_jsonmode": {k: no_jm[k] for k in ("n", "f1", "ci", "parse_failures")},
        "delta": deltas,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("\n--- Zusammenfassung (Gesamt-F1) ---")
    print(f"  full:         {full['f1']['GESAMT']:.4f}")
    for name in ("no_whitelist", "no_regex", "no_jsonmode"):
        dd = deltas[name]
        print(f"  {name:<13} F1_ohne={dd['f1_ohne']:.4f}  "
              f"deltaF1={dd['delta_full_minus_ohne']:+.4f}  "
              f"[{dd['ci_lo']:+.4f}; {dd['ci_hi']:+.4f}]")
    print(f"\nGespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
