"""
Erweitertes Robustheitsexperiment auf n=50 IMDb-Filmen mit
Bootstrap-Konfidenzintervallen für den feldweisen F1-Score und für
die Differenz ΔF1 = F1(LLM) − F1(XPath) je Mutationsstufe.

Outputs:
    robustness_results_n50.json  - vollständige Tabelle inkl. CIs

Vergleich gegen die ursprüngliche n=20-Auswertung wird in
robustness_compare_n20_n50.py gerechnet (separater Schritt).
"""
import sys
import json
import re
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from bs4 import BeautifulSoup
from scraper.cleaner import HTMLCleaner
from llm.ollama_client import OllamaClient

GT_PATH = Path(__file__).parent.parent / "evaluation" / "ground_truth_250.json"
MUT_DIR = Path(__file__).parent / "mutations"
OUT_PATH = Path(__file__).parent / "robustness_results_n50.json"

LEVELS = ["M0", "M1", "M2", "M3", "M4"]
FIELDS = ["titel", "jahr", "bewertung", "dauer_minuten", "regisseur", "genres"]

BOOTSTRAP_N = 10000
CI_ALPHA = 0.05
SEED = 42


# ---------- XPath/CSS-Baseline (deterministisch, bindet an DOM) ----------

def xpath_extract(html: str) -> dict:
    """Klassischer Extraktor mit fest verdrahteten CSS-Selektoren."""
    soup = BeautifulSoup(html, "lxml")
    out = {f: None for f in FIELDS}

    t = soup.select_one('[data-testid="hero__pageTitle"]')
    if t:
        out["titel"] = t.get_text(strip=True)

    y = soup.select_one(
        '[data-testid="title-details-releasedate"] '
        '.ipc-metadata-list-item__list-content-item'
    )
    if y:
        m = re.search(r'\b(19|20)\d{2}\b', y.get_text())
        if m:
            out["jahr"] = int(m.group(0))

    r = soup.select_one('[data-testid="hero-rating-bar__aggregate-rating__score"]')
    if r:
        try:
            out["bewertung"] = float(re.search(r'\d+\.\d+', r.get_text()).group(0))
        except Exception:
            pass

    rt = soup.select_one(
        '[data-testid="title-techspec_runtime"] '
        '.ipc-metadata-list-item__content-container'
    )
    if rt:
        rt_text = rt.get_text(strip=True)
        mh = re.search(r'(\d+)\s*h\s*(\d+)\s*m', rt_text)
        if mh:
            out["dauer_minuten"] = int(mh.group(1)) * 60 + int(mh.group(2))
        else:
            mh = re.search(r'(\d+)\s*h', rt_text)
            mm = re.search(r'(\d+)\s*m', rt_text)
            tot = 0
            if mh:
                tot += int(mh.group(1)) * 60
            if mm:
                tot += int(mm.group(1))
            if tot > 0:
                out["dauer_minuten"] = tot

    chips = soup.select('.ipc-chip__text')
    if chips:
        VALID = {
            "Action","Adventure","Animation","Biography","Comedy","Crime",
            "Documentary","Drama","Family","Fantasy","Film-Noir","History",
            "Horror","Music","Musical","Mystery","Romance","Sci-Fi","Short",
            "Sport","Thriller","War","Western",
        }
        out["genres"] = [c.get_text(strip=True) for c in chips
                         if c.get_text(strip=True) in VALID]

    d = soup.select('[data-testid="title-pc-principal-credit"]:first-of-type a')
    if d:
        out["regisseur"] = ", ".join(x.get_text(strip=True) for x in d)

    return out


# ---------- LLM-Pipeline ----------

def llm_extract(html: str, cleaner: HTMLCleaner, client: OllamaClient) -> dict:
    cleaned = cleaner.clean_for_imdb(html)
    if len(cleaned) < 100:
        cleaned = cleaner.clean(html)
    data = client.extract_film_data(cleaned, use_few_shot=True)
    if not data:
        return {f: None for f in FIELDS}
    return {f: data.get(f) for f in FIELDS}


# ---------- Match-Logik ----------

def normalize_string(s):
    if not s:
        return ""
    return re.sub(r'\s+', ' ', str(s).strip().lower())


def match(field, pred, exp):
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


def evaluate_extractor(extractor_fn, level, gt_by_id, label, **kwargs):
    """Wendet Extraktor auf alle Snapshots einer Mutationsstufe an."""
    snap_dir = MUT_DIR / level
    matches = {f: [] for f in FIELDS}  # je Film 0/1
    ids = []
    for snap_file in sorted(snap_dir.glob("*.html")):
        imdb_id = snap_file.stem
        if imdb_id not in gt_by_id:
            continue
        ids.append(imdb_id)
        html = snap_file.read_text(encoding="utf-8")
        try:
            pred = extractor_fn(html, **kwargs)
        except Exception as e:
            print(f"  [{label}/{level}] {imdb_id} ERROR: {e}")
            pred = {f: None for f in FIELDS}
        for f in FIELDS:
            matches[f].append(1 if match(f, pred.get(f), gt_by_id[imdb_id].get(f)) else 0)
    n = len(ids)
    f1 = {f: float(np.mean(matches[f])) for f in FIELDS} if n else {}
    if n:
        all_scores = [matches[f][i] for f in FIELDS for i in range(n)]
        f1["GESAMT"] = float(np.mean(all_scores))
    else:
        f1["GESAMT"] = 0.0
    print(f"  [{label}/{level}] {n} Filme, F1_gesamt={f1['GESAMT']:.3f}")
    return {"n": n, "f1": f1, "matches": matches, "ids": ids}


def bootstrap_ci(values, alpha=CI_ALPHA, n_boot=BOOTSTRAP_N, seed=SEED):
    """Klassische Bootstrap-Perzentil-CI für den Mittelwert."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0)
    boots = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (lo, hi)


def paired_bootstrap_diff_ci(a, b, alpha=CI_ALPHA, n_boot=BOOTSTRAP_N, seed=SEED):
    """Gepaarte Bootstrap-CI für die Differenz mean(a) − mean(b).

    a und b sind 0/1-Listen gleicher Länge (gleiche Filme).
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b), "gepaarte Bootstrap erwartet gleiche Länge"
    n = len(a)
    if n == 0:
        return (0.0, 0.0, 0.0)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(np.mean(a[idx]) - np.mean(b[idx]))
    diffs = np.asarray(diffs)
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    point = float(np.mean(a) - np.mean(b))
    return (point, lo, hi)


def attach_cis(per_extractor):
    """Fügt jedem Eintrag (per Mutationsstufe) feldweise CIs hinzu."""
    for level, entry in per_extractor.items():
        cis = {}
        for f in FIELDS:
            cis[f] = bootstrap_ci(entry["matches"][f])
        all_scores = [entry["matches"][f][i]
                      for f in FIELDS
                      for i in range(entry["n"])]
        cis["GESAMT"] = bootstrap_ci(all_scores)
        entry["ci"] = {k: {"lo": round(v[0], 4), "hi": round(v[1], 4)} for k, v in cis.items()}
        entry["f1"] = {k: round(v, 4) for k, v in entry["f1"].items()}


def compute_delta(llm_block, xpath_block):
    """ΔF1 = F1(LLM) − F1(XPath) auf gleicher Filmmenge je Stufe."""
    deltas = {}
    for level in LEVELS:
        L = llm_block[level]
        X = xpath_block[level]
        # Schnittmenge der IDs (sollte identisch sein, defensiv)
        common = [i for i in L["ids"] if i in X["ids"]]
        if not common:
            continue
        l_idx = {i: k for k, i in enumerate(L["ids"])}
        x_idx = {i: k for k, i in enumerate(X["ids"])}
        delta_per_field = {}
        for f in FIELDS + ["GESAMT"]:
            if f == "GESAMT":
                a = [m for i in common for m in [L["matches"][g][l_idx[i]] for g in FIELDS]]
                b = [m for i in common for m in [X["matches"][g][x_idx[i]] for g in FIELDS]]
            else:
                a = [L["matches"][f][l_idx[i]] for i in common]
                b = [X["matches"][f][x_idx[i]] for i in common]
            point, lo, hi = paired_bootstrap_diff_ci(a, b)
            delta_per_field[f] = {
                "delta": round(point, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "n": len(common),
            }
        deltas[level] = delta_per_field
    return deltas


def main():
    with open(GT_PATH) as f:
        gt = json.load(f)
    gt_by_id = {x["imdb_id"]: x for x in gt["filme"]}

    cleaner = HTMLCleaner()
    client = OllamaClient(model="qwen2.5:3b")

    if not client.is_available():
        print("Ollama nicht erreichbar oder Modell nicht geladen.")
        sys.exit(1)

    results = {
        "xpath": {}, "llm": {},
        "meta": {
            "n_films": 50,
            "model": "qwen2.5:3b",
            "levels": LEVELS,
            "bootstrap_n": BOOTSTRAP_N,
            "ci_alpha": CI_ALPHA,
            "description": "Erweiterte Robustheitsstudie n=50 mit Bootstrap-CIs",
        }
    }

    print("=== XPath-Baseline ===")
    for level in LEVELS:
        results["xpath"][level] = evaluate_extractor(
            lambda html: xpath_extract(html), level, gt_by_id, "xpath")

    print("\n=== LLM-Pipeline (qwen2.5:3b) ===")
    for level in LEVELS:
        t0 = time.time()
        results["llm"][level] = evaluate_extractor(
            lambda html, c=cleaner, cl=client: llm_extract(html, c, cl),
            level, gt_by_id, "llm")
        results["llm"][level]["seconds"] = round(time.time() - t0, 1)

    # Bootstrap-CIs anhängen
    attach_cis(results["xpath"])
    attach_cis(results["llm"])

    # ΔF1 mit gepaartem Bootstrap
    results["delta"] = compute_delta(results["llm"], results["xpath"])

    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nGespeichert: {OUT_PATH}")

    # Druckbare Zusammenfassung
    print("\n--- ΔF1 (LLM − XPath) ---")
    print(f"{'Stufe':<6} {'F1_LLM':>8} {'F1_XP':>8} {'Δ':>8} {'CI_lo':>8} {'CI_hi':>8}")
    for level in LEVELS:
        L = results["llm"][level]["f1"]["GESAMT"]
        X = results["xpath"][level]["f1"]["GESAMT"]
        d = results["delta"][level]["GESAMT"]
        print(f"{level:<6} {L:>8.3f} {X:>8.3f} {d['delta']:>+8.3f} "
              f"{d['ci_lo']:>+8.3f} {d['ci_hi']:>+8.3f}")


if __name__ == "__main__":
    main()
