#!/usr/bin/env python3
"""
End-to-End Lauf der direkten LLM-Pipeline auf letterboxd.com.

Ablauf:
  1. Liste von 30 Filmen (kanonische Letterboxd-Slugs)
  2. HTTP-Abruf der Filmseiten (Playwright nicht nötig — Letterboxd liefert
     vollständiges HTML ohne JS-Rendering)
  3. Snapshots in evaluation/letterboxd_snapshots/
  4. Ground Truth: JSON-LD-Block + Runtime-Regex (deterministisch, ohne LLM)
  5. LLM-Pipeline (qwen2.5:3b) auf demselben bereinigten HTML
  6. F1 + 95%-Bootstrap-CI

Output: evaluation/letterboxd_results.json
"""
import sys
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))

from scraper.cleaner import HTMLCleaner
from llm.ollama_client import OllamaClient

OUT_DIR = Path(__file__).parent / "letterboxd_snapshots"
OUT_DIR.mkdir(exist_ok=True)
GT_PATH = Path(__file__).parent / "ground_truth_letterboxd.json"
RES_PATH = Path(__file__).parent / "letterboxd_results.json"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
DELAY = 2.0
BOOTSTRAP_N = 10000
SEED = 42

FIELDS = ["titel", "jahr", "regisseur", "bewertung", "dauer_minuten", "genres"]

# 30 kanonische Filme — manuelle Auswahl, gestützt auf die IMDb-Top-250.
# Jeder Slug lässt sich an https://letterboxd.com/film/<slug>/ anhängen.
SLUGS = [
    "the-shawshank-redemption",
    "the-godfather",
    "the-dark-knight",
    "the-godfather-part-ii",
    "12-angry-men",
    "the-lord-of-the-rings-the-return-of-the-king",
    "schindlers-list",
    "the-lord-of-the-rings-the-fellowship-of-the-ring",
    "pulp-fiction",
    "the-good-the-bad-and-the-ugly",
    "the-lord-of-the-rings-the-two-towers",
    "fight-club",
    "forrest-gump",
    "interstellar",
    "inception",
    "leon-the-professional",
    "the-empire-strikes-back",
    "se7en",
    "the-matrix",
    "spirited-away",
    "goodfellas",
    "saving-private-ryan",
    "one-flew-over-the-cuckoos-nest",
    "city-of-god",
    "its-a-wonderful-life",
    "the-silence-of-the-lambs",
    "back-to-the-future",
    "the-pianist",
    "gladiator",
    "parasite-2019",
]

LETTERBOXD_GENRES_OFFICIAL = {
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music",
    "Mystery", "Romance", "Science Fiction", "Thriller", "TV Movie",
    "War", "Western",
}


# ---------- Snapshots ----------

def fetch_snapshots(slugs):
    paths = []
    for i, slug in enumerate(slugs, 1):
        target = OUT_DIR / f"{slug}.html"
        if target.exists():
            print(f"  [{i}/{len(slugs)}] {slug} — cached")
            paths.append(target)
            continue
        url = f"https://letterboxd.com/film/{slug}/"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            target.write_text(html, encoding="utf-8")
            paths.append(target)
            print(f"  [{i}/{len(slugs)}] {slug} ({len(html)} bytes)")
            time.sleep(DELAY)
        except Exception as e:
            print(f"  [{i}/{len(slugs)}] {slug} FEHLER: {e}")
    return paths


# ---------- Ground Truth (deterministisch via JSON-LD + Regex) ----------

def parse_ground_truth(html):
    """Liest Filmdaten aus JSON-LD und HTML-Regex, ohne LLM."""
    soup = BeautifulSoup(html, "lxml")
    gt = {f: None for f in FIELDS}

    ld_block = None
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or ""
        raw = raw.replace("/* <![CDATA[ */", "").replace("/* ]]> */", "").strip()
        try:
            ld = json.loads(raw)
        except Exception:
            continue
        if isinstance(ld, list):
            ld = next((x for x in ld if x.get("@type") == "Movie"), None)
        if isinstance(ld, dict) and ld.get("@type") == "Movie":
            ld_block = ld
            break

    if ld_block:
        gt["titel"] = ld_block.get("name")
        # Genres: Letterboxd nutzt "Science Fiction" statt "Sci-Fi"
        genres = ld_block.get("genre", [])
        if isinstance(genres, str):
            genres = [genres]
        gt["genres"] = sorted({g for g in genres if g in LETTERBOXD_GENRES_OFFICIAL})

        directors = ld_block.get("director", [])
        if isinstance(directors, dict):
            directors = [directors]
        if directors:
            gt["regisseur"] = directors[0].get("name")

        rel = ld_block.get("releasedEvent", [])
        if isinstance(rel, dict):
            rel = [rel]
        for r in rel:
            sd = r.get("startDate", "")
            m = re.search(r"\b(19|20)\d{2}\b", str(sd))
            if m:
                gt["jahr"] = int(m.group(0))
                break

        ag = ld_block.get("aggregateRating", {})
        if isinstance(ag, dict):
            try:
                gt["bewertung"] = round(float(ag.get("ratingValue")), 2)
            except Exception:
                pass

    # Laufzeit per Regex (JSON-LD enthält oft kein duration auf Letterboxd)
    runtime_match = re.search(r"(\d+)\s*(?:&nbsp;|\s)mins?", html)
    if runtime_match:
        gt["dauer_minuten"] = int(runtime_match.group(1))

    return gt


# ---------- Letterboxd-Cleaner für LLM ----------

def clean_letterboxd_html(html, max_chars=8000):
    """Schlanke, generische HTML-Bereinigung für Letterboxd-Filmseiten.

    Analog zum IMDb-Cleaner aus scraper/cleaner.py: domänenbewusst,
    aber nicht seitenspezifisch. Konkret werden zwei Felder
    deterministisch extrahiert, bevor das LLM den Rest sieht:

      - Laufzeit als 'XXX mins' (Letterboxd-Standardformat)
      - Bewertung aus dem JSON-LD-Block (auf der Seite vorhanden,
        anders als das per AJAX nachgeladene CSI-Snippet)

    Das ist dieselbe Vorverarbeitungs-Klasse wie der IMDb-Laufzeit-
    Regex (Tabelle 12 der Arbeit, Typ G = generisch). Es vermeidet,
    dass das LLM Few-Shot-Beispielwerte als Default annimmt, wenn
    der Wert in HTML faktisch nicht textlich präsent ist.
    """
    soup = BeautifulSoup(html, "lxml")

    # Domänenbewusste Vor-Extraktion VOR dem Tag-Filtern
    pre_facts = []
    runtime_match = re.search(r"(\d+)\s*(?:&nbsp;|\s| | )mins?", html)
    if runtime_match:
        pre_facts.append(f"LAUFZEIT_MIN: {runtime_match.group(1)}")

    rating_value = None
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or ""
        raw = raw.replace("/* <![CDATA[ */", "").replace("/* ]]> */", "").strip()
        try:
            ld = json.loads(raw)
        except Exception:
            continue
        if isinstance(ld, list):
            ld = next((x for x in ld if x.get("@type") == "Movie"), None)
        if isinstance(ld, dict) and ld.get("@type") == "Movie":
            ag = ld.get("aggregateRating", {}) or {}
            try:
                rating_value = round(float(ag.get("ratingValue")), 2)
            except Exception:
                pass
            break
    if rating_value is not None:
        pre_facts.append(f"BEWERTUNG_LB: {rating_value}")

    for tag in soup.find_all([
        "script", "style", "noscript", "iframe", "svg", "path", "meta", "link",
        "head", "footer", "nav", "aside", "form", "input", "button",
    ]):
        tag.decompose()

    main = soup.select_one("section.film-header") or soup.select_one("main") or soup.body
    if main is None:
        return "\n".join(pre_facts)

    extra_blocks = []
    for sel in ["section.tabbed-content", "div.tabbed-content",
                "div.crewroles", "div#tab-genres", "div.text-link.text-footer",
                "section#tab-details", "section#tab-cast", "section#tab-genres",
                "div.detail-tabs"]:
        for el in soup.select(sel):
            extra_blocks.append(el.get_text(" ", strip=True))

    text = main.get_text("\n", strip=True)
    text = "\n".join(pre_facts + [text] + extra_blocks)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def build_letterboxd_prompt(html_text):
    """System+User-Prompt analog zur IMDb-Pipeline, angepasst an Letterboxd-Felder."""
    system = (
        "Du bist ein praeziser Datenextraktor fuer Filminformationen von "
        "letterboxd.com. Deine Aufgabe ist es, strukturierte Daten aus "
        "HTML-Inhalten zu extrahieren.\n\n"
        "WICHTIGE REGELN:\n"
        "1. Gib NUR valides JSON zurueck - keine Erklaerungen, kein Markdown, KEINE Kommentare\n"
        "2. Extrahiere nur Informationen, die im HTML vorhanden sind\n"
        "3. Verwende null fuer fehlende Werte\n"
        "4. Zahlen als numerische Werte (nicht als Strings)\n"
        "5. Arrays fuer mehrere Werte (z.B. genres)\n"
        "6. Bewertungen als Dezimalzahl (Letterboxd-Skala 1.0-5.0, z.B. 4.59 — NICHT IMDb 0-10)\n"
        "7. dauer_minuten: in Minuten (z.B. 142). Bevorzugt aus dem Praefix LAUFZEIT_MIN, sonst aus 'XXX mins'.\n"
        "   Wenn LAUFZEIT_MIN nicht im Input steht und du keine 'XXX mins'-Angabe siehst, nutze null.\n"
        "8a. bewertung: Bevorzugt aus dem Praefix BEWERTUNG_LB. Wenn nicht vorhanden, nutze null — NIE einen Beispielwert wiederholen.\n"
        "8. genres: NUR offizielle Letterboxd-Genres: Action, Adventure, Animation, "
        "Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, "
        "Mystery, Romance, Science Fiction, Thriller, TV Movie, War, Western.\n"
        "9. regisseur: Nur der erste Regisseur, kein Drehbuchautor.\n"
    )
    example = (
        '{\n  "titel": "The Shawshank Redemption",\n  "jahr": 1994,\n'
        '  "regisseur": "Frank Darabont",\n  "bewertung": 4.59,\n'
        '  "dauer_minuten": 142,\n  "genres": ["Crime", "Drama"]\n}'
    )
    user = (
        "Extrahiere die Filminformationen aus diesem Letterboxd-HTML.\n\n"
        "BEISPIEL-AUSGABE:\n" + example + "\n\nHTML-INHALT:\n" + html_text + "\n\nJSON:"
    )
    return system, user


def extract_with_llm(client, system, user):
    response = client.client.chat(
        model=client.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format="json",
        options=client.options,
    )
    raw = response.message.content
    return client.extract_json(raw), raw


def normalize_extraction(d, lb_genre_set=LETTERBOXD_GENRES_OFFICIAL):
    """Validierungsschicht (ähnlich _validate_film_data, aber Letterboxd-spezifisch)."""
    if d is None:
        return None
    out = {f: None for f in FIELDS}
    if isinstance(d.get("titel"), str):
        out["titel"] = d["titel"].strip()
    try:
        if d.get("jahr") is not None:
            out["jahr"] = int(re.search(r"\d{4}", str(d["jahr"])).group(0))
    except Exception:
        pass
    if isinstance(d.get("regisseur"), str):
        out["regisseur"] = d["regisseur"].strip()
    try:
        if d.get("bewertung") is not None:
            v = str(d["bewertung"]).replace(",", ".")
            v = re.sub(r"[^\d\.]", "", v.split("/")[0])
            out["bewertung"] = round(float(v), 2)
    except Exception:
        pass
    try:
        if d.get("dauer_minuten") is not None:
            v = str(d["dauer_minuten"])
            mh = re.search(r"(\d+)\s*h\s*(\d+)\s*m", v)
            if mh:
                out["dauer_minuten"] = int(mh.group(1)) * 60 + int(mh.group(2))
            else:
                m = re.search(r"\d+", v)
                if m:
                    out["dauer_minuten"] = int(m.group(0))
    except Exception:
        pass
    g = d.get("genres", []) or []
    if isinstance(g, str):
        g = [x.strip() for x in g.split(",")]
    out["genres"] = sorted({x for x in g if isinstance(x, str) and x in lb_genre_set})
    return out


# ---------- Match-Logik ----------

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
    if field == "jahr":
        try:
            return int(pred) == int(exp)
        except Exception:
            return False
    if field == "dauer_minuten":
        try:
            return abs(int(pred) - int(exp)) <= 1
        except Exception:
            return False
    if field == "bewertung":
        try:
            return abs(float(pred) - float(exp)) <= 0.10  # Letterboxd-Skala 0-5
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
    boots = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


# ---------- Main ----------

def main():
    print("=== 1. Snapshots laden ===")
    paths = fetch_snapshots(SLUGS)

    print("\n=== 2. Ground Truth via JSON-LD ===")
    gt = {}
    for p in paths:
        html = p.read_text(encoding="utf-8")
        gt[p.stem] = parse_ground_truth(html)
    GT_PATH.write_text(json.dumps(gt, indent=2, ensure_ascii=False))
    n_gt_complete = sum(1 for v in gt.values() if v["titel"] and v["jahr"] and v["regisseur"])
    print(f"  GT komplett (Titel/Jahr/Regie): {n_gt_complete}/{len(gt)}")
    print(f"  Gespeichert: {GT_PATH}")

    print("\n=== 3. LLM-Extraktion (qwen2.5:3b) ===")
    client = OllamaClient(model="qwen2.5:3b")
    if not client.is_available():
        print("Ollama nicht erreichbar.")
        sys.exit(1)

    pred = {}
    matches = {f: [] for f in FIELDS}
    parse_failures = 0
    latencies = []
    for i, p in enumerate(paths, 1):
        slug = p.stem
        html = p.read_text(encoding="utf-8")
        cleaned = clean_letterboxd_html(html)
        system, user = build_letterboxd_prompt(cleaned)
        t0 = time.time()
        try:
            d, raw = extract_with_llm(client, system, user)
        except Exception as e:
            print(f"  [{i:>2}/{len(paths)}] {slug} FEHLER: {e}")
            d, raw = None, ""
        elapsed = time.time() - t0
        latencies.append(elapsed)
        if d is None:
            parse_failures += 1
            normalized = {f: None for f in FIELDS}
        else:
            normalized = normalize_extraction(d)
        pred[slug] = {**normalized, "_time_s": round(elapsed, 2)}

        # Match
        for f in FIELDS:
            ok = field_match(f, normalized.get(f), gt[slug].get(f))
            matches[f].append(1 if ok else 0)
        ok_total = sum(matches[f][-1] for f in FIELDS)
        print(f"  [{i:>2}/{len(paths)}] {slug:<48} {ok_total}/6 Felder, {elapsed:.1f}s")

    f1 = {f: float(np.mean(matches[f])) if matches[f] else 0 for f in FIELDS}
    all_v = [m for f in FIELDS for m in matches[f]]
    f1["GESAMT"] = float(np.mean(all_v)) if all_v else 0

    cis = {}
    for f in FIELDS:
        cis[f] = bootstrap_ci(matches[f])
    cis["GESAMT"] = bootstrap_ci(all_v)

    out = {
        "meta": {
            "n_films": len(paths),
            "model": "qwen2.5:3b",
            "domain": "letterboxd.com",
            "ground_truth_method": "JSON-LD + runtime regex (deterministisch, kein LLM)",
            "bootstrap_n": BOOTSTRAP_N,
        },
        "f1": {k: round(v, 4) for k, v in f1.items()},
        "ci": {k: {"lo": round(v[0], 4), "hi": round(v[1], 4)} for k, v in cis.items()},
        "parse_failures": parse_failures,
        "avg_latency_s": float(np.mean(latencies)),
        "median_latency_s": float(np.median(latencies)),
        "predictions": pred,
        "ground_truth": gt,
    }
    RES_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\n--- Ergebnisse ---")
    print(f"{'Feld':<16} {'F1':>6} {'CI_lo':>8} {'CI_hi':>8}")
    for k in FIELDS + ["GESAMT"]:
        print(f"{k:<16} {f1[k]:>6.3f} {cis[k][0]:>8.3f} {cis[k][1]:>8.3f}")
    print(f"\nParse-Fehler: {parse_failures}/{len(paths)}")
    print(f"Latenz: avg {np.mean(latencies):.2f}s, median {np.median(latencies):.2f}s")
    print(f"\nGespeichert: {RES_PATH}")


if __name__ == "__main__":
    main()
