"""
Evaluiert die direkte LLM-Pipeline und einen klassischen XPath-Baseline-
Extraktor auf den 5 Mutationsstufen (M0-M4).

Vergleichsmetrik: feldweiser F1-Score gegen Ground Truth aus
ground_truth_250.json. Ergebnisse werden in robustness_results.json
gespeichert.

Zwei Extraktoren:
  - LLM-Pipeline (HTMLCleaner -> qwen2.5:3b mit JSON-Modus)
  - XPath-Baseline (CSS-Selektor-basierter Parser, der genau die
    DOM-Pfade der Originalseite kennt)
"""
import sys
import json
import re
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bs4 import BeautifulSoup
from scraper.cleaner import HTMLCleaner
from llm.ollama_client import OllamaClient

GT_PATH = Path(__file__).parent.parent / "evaluation" / "ground_truth_250.json"
MUT_DIR = Path(__file__).parent / "mutations"
OUT_PATH = Path(__file__).parent / "robustness_results.json"

LEVELS = ["M0", "M1", "M2", "M3", "M4"]
FIELDS = ["titel", "jahr", "bewertung", "dauer_minuten", "regisseur", "genres"]


# ---------- XPath/CSS-Baseline (deterministisch, bindet an DOM) ----------

def xpath_extract(html: str) -> dict:
    """Klassischer Extraktor mit fest verdrahteten CSS-Selektoren.

    Bindet sich an die exakte IMDb-DOM-Struktur (data-testid + class).
    Bricht bei Layout-Mutationen, weil die Selektoren keine Fallbacks haben.
    """
    soup = BeautifulSoup(html, "lxml")
    out = {f: None for f in FIELDS}

    # Titel: data-testid="hero__pageTitle"
    t = soup.select_one('[data-testid="hero__pageTitle"]')
    if t:
        out["titel"] = t.get_text(strip=True)

    # Jahr: data-testid="title-details-releasedate"
    y = soup.select_one(
        '[data-testid="title-details-releasedate"] '
        '.ipc-metadata-list-item__list-content-item'
    )
    if y:
        m = re.search(r'\b(19|20)\d{2}\b', y.get_text())
        if m:
            out["jahr"] = int(m.group(0))

    # Bewertung
    r = soup.select_one('[data-testid="hero-rating-bar__aggregate-rating__score"]')
    if r:
        try:
            out["bewertung"] = float(re.search(r'\d+\.\d+', r.get_text()).group(0))
        except Exception:
            pass

    # Laufzeit
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

    # Genres: class ipc-chip__text
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

    # Regisseur: data-testid="title-pc-principal-credit"
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


# ---------- Match-Logik (gleich wie evaluate_250.py) ----------

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
    """Wendet Extraktor auf alle 20 Snapshots einer Mutationsstufe an."""
    snap_dir = MUT_DIR / level
    matches = {f: 0 for f in FIELDS}
    n = 0
    for snap_file in sorted(snap_dir.glob("*.html")):
        imdb_id = snap_file.stem
        if imdb_id not in gt_by_id:
            continue
        n += 1
        html = snap_file.read_text(encoding="utf-8")
        try:
            pred = extractor_fn(html, **kwargs)
        except Exception as e:
            print(f"  [{label}/{level}] {imdb_id} ERROR: {e}")
            pred = {f: None for f in FIELDS}
        for f in FIELDS:
            if match(f, pred.get(f), gt_by_id[imdb_id].get(f)):
                matches[f] += 1
    f1 = {f: matches[f] / n for f in FIELDS} if n else {}
    f1["GESAMT"] = sum(matches.values()) / (n * len(FIELDS)) if n else 0
    print(f"  [{label}/{level}] {n} Filme, F1_gesamt={f1['GESAMT']:.3f}")
    return {"n": n, "f1": f1, "matches": matches}


def main():
    with open(GT_PATH) as f:
        gt = json.load(f)
    gt_by_id = {x["imdb_id"]: x for x in gt["filme"]}

    cleaner = HTMLCleaner()
    client = OllamaClient(model="qwen2.5:3b")

    if not client.is_available():
        print("Ollama nicht erreichbar oder Modell nicht geladen.")
        sys.exit(1)

    results = {"xpath": {}, "llm": {}, "meta": {
        "n_films": 20, "model": "qwen2.5:3b",
        "levels": LEVELS,
        "description": "Robustheit gegen Layout-Mutation"
    }}

    print("=== XPath-Baseline ===")
    for level in LEVELS:
        results["xpath"][level] = evaluate_extractor(
            lambda html: xpath_extract(html),
            level, gt_by_id, "xpath"
        )

    print("\n=== LLM-Pipeline (qwen2.5:3b) ===")
    for level in LEVELS:
        t0 = time.time()
        results["llm"][level] = evaluate_extractor(
            lambda html, c=cleaner, cl=client: llm_extract(html, c, cl),
            level, gt_by_id, "llm"
        )
        results["llm"][level]["seconds"] = round(time.time() - t0, 1)

    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nGespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
