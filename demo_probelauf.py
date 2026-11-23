#!/usr/bin/env python3
"""
Probelauf fürs Kolloquium.

Gleiche Pipeline wie main.py (clean_for_imdb -> qwen2.5:3b -> Validierung),
aber auf gespeicherten Original-IMDb-Seiten (robustness/mutations/M0).
Grund: IMDb blockt automatisierte Zugriffe derzeit mit einer
"Human Verification"-Seite (Stand 29.09.2026), der Live-Lauf findet dann
keine Filme.

Verwendung:
    python demo_probelauf.py               # 5 Filme (Rang 1-5)
    python demo_probelauf.py -n 3          # 3 Filme
    python demo_probelauf.py --eingabe     # zusätzlich: was das Modell bekommt
    python demo_probelauf.py --redesign    # zusätzlich: XPath vs. LLM auf M4
    python demo_probelauf.py --wiederholen # ersten Film 3x extrahieren: gleiche Ausgabe?
    python demo_probelauf.py --anzeigen output/imdb_top_5_XXXX.json
                                           # Ergebnis eines Live-Laufs (main.py) als Tabelle
"""
import argparse
import importlib.util
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import OLLAMA_CONFIG, OUTPUT_DIR
from scraper.cleaner import HTMLCleaner
from llm.ollama_client import OllamaClient

SNAP_DIR = ROOT / "robustness" / "mutations"
GT_PATH = ROOT / "evaluation" / "ground_truth_250.json"
FIELDS = ["titel", "jahr", "dauer_minuten", "bewertung", "regisseur", "genres"]
LABELS = {"titel": "Titel", "jahr": "Jahr", "dauer_minuten": "Laufzeit",
          "bewertung": "Bewertung", "regisseur": "Regie", "genres": "Genres"}

console = Console()
logging.disable(logging.WARNING)  # keine Log-Warnungen in der Demo

# Match-Logik und XPath-Baseline genau wie in der Evaluation der Arbeit
_spec = importlib.util.spec_from_file_location(
    "robustness_eval", ROOT / "robustness" / "03_evaluate_n50.py")
_rob = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rob)
match = _rob.match
xpath_extract = _rob.xpath_extract


def de(x):
    """Zahl mit deutschem Dezimalkomma bzw. Tausenderpunkt."""
    if isinstance(x, float):
        return f"{x:.1f}".replace(".", ",")
    return f"{x:,}".replace(",", ".")


def fmt(field, value):
    if value is None:
        return "null"
    if field == "genres":
        return ", ".join(value) if value else "[]"
    if field == "dauer_minuten":
        return f"{value} min"
    return str(value)


def llm_extract(html, cleaner, client):
    """Wie die Pipeline: IMDb-Cleaner, bei zu wenig Text generischer Cleaner."""
    cleaned = cleaner.clean_for_imdb(html)
    if len(cleaned) < 100:
        cleaned = cleaner.clean(html)
    data = client.extract_film_data(cleaned, use_few_shot=True)
    return cleaned, data or {}


def print_results(results):
    """Tabelle: grün = stimmt mit Referenz, rot = weicht ab (Referenz in Klammern)."""
    table = Table(title="Was das Modell extrahiert hat  (grün = stimmt, rot = weicht ab, Referenz in Klammern)",
                  show_lines=True)
    for f in FIELDS:
        table.add_column(LABELS[f], overflow="fold")
    ok = 0
    for r in results:
        row = []
        for f in FIELDS:
            pred, exp = r["data"].get(f), r["gt"].get(f) if r["gt"] else None
            if r["gt"] is None:
                row.append(fmt(f, pred))
            elif match(f, pred, exp):
                ok += 1
                row.append(f"[green]{fmt(f, pred)}[/green]")
            else:
                row.append(f"[red]{fmt(f, pred)}[/red]\n[dim]({fmt(f, exp)})[/dim]")
        table.add_row(*row)
    console.print(table)
    n = sum(1 for r in results if r["gt"]) * len(FIELDS)
    console.print(f"[bold]{ok} von {n} Feldern richtig[/bold]")


def print_cast(titel, cast):
    """m:n-Beziehung Film <-> Person, Rolle als Attribut der Verbindung."""
    if not cast:
        return
    ct = Table(title=f"Besetzung als m:n-Beziehung (actors2movies): {titel}")
    ct.add_column("Schauspieler")
    ct.add_column("Rolle (Attribut der Verbindung)")
    for c in cast[:5]:
        ct.add_row(c.get("name") or "", c.get("als_charakter") or "")
    if len(cast) > 5:
        ct.add_row(f"[dim]… {len(cast) - 5} weitere[/dim]", "")
    console.print(ct)


def show_live_file(path, gt_by_id):
    """Ergebnis eines echten Live-Laufs (main.py) anzeigen."""
    d = json.load(open(path, encoding="utf-8"))
    movies = d.get("movies", [])
    if not movies:
        console.print("[red]In der Datei sind keine Filme.[/red]")
        return 1
    console.print(Panel(f"Live-Lauf vom {d['metadata'].get('scraped_at', '?')[:16]}\n"
                        f"Modell: {d['metadata'].get('llm_model')}  ·  {len(movies)} Filme",
                        title="Ergebnis aus main.py", border_style="cyan"))
    results = [{"gt": gt_by_id.get(m.get("imdb_id")), "data": m} for m in movies]
    print_results(results)
    raw = movies[0].get("raw_json") or {}
    print_cast(movies[0].get("titel"), raw.get("besetzung"))
    return 0


def main():
    ap = argparse.ArgumentParser(description="Probelauf auf gespeicherten IMDb-Seiten")
    ap.add_argument("-n", type=int, default=5, help="Anzahl Filme (Rang 1..n), Standard 5")
    ap.add_argument("--eingabe", action="store_true", help="zeigt, was das Modell für den ersten Film bekommt")
    ap.add_argument("--redesign", action="store_true", help="vergleicht XPath und LLM auf M4 (alle Selektoren kaputt)")
    ap.add_argument("--wiederholen", action="store_true", help="ersten Film 3x extrahieren und vergleichen")
    ap.add_argument("--anzeigen", metavar="JSON", help="Ergebnisdatei eines Live-Laufs anzeigen")
    args = ap.parse_args()

    gt = json.load(open(GT_PATH, encoding="utf-8"))["filme"]
    gt_by_id = {f["imdb_id"]: f for f in gt}
    if args.anzeigen:
        return show_live_file(args.anzeigen, gt_by_id)

    available = {p.stem for p in (SNAP_DIR / "M0").glob("*.html")}
    films = [f for f in sorted(gt, key=lambda x: x["rang"]) if f["imdb_id"] in available][: args.n]

    cleaner = HTMLCleaner()
    client = OllamaClient()
    if not client.is_available():
        console.print("[red]Ollama ist nicht erreichbar. Erst in einem zweiten Terminal 'ollama serve' starten.[/red]")
        return 1

    temp = str(OLLAMA_CONFIG["options"]["temperature"]).replace(".", ",")
    console.print(Panel(
        f"[bold]Probelauf: {len(films)} Filme[/bold]\n"
        f"Modell: {client.model} (lokal über Ollama), Temperatur {temp}\n"
        f"Eingabe: gespeicherte Original-IMDb-Seiten (je ca. 1,3 MB HTML)\n"
        f"Ablauf: HTML-Cleaner  →  LLM  →  JSON  →  Abgleich mit Referenz",
        title="IMDb-Pipeline", border_style="cyan"))

    results = []
    for i, film in enumerate(films, 1):
        html = (SNAP_DIR / "M0" / f"{film['imdb_id']}.html").read_text(encoding="utf-8")
        t0 = time.time()
        cleaned, data = llm_extract(html, cleaner, client)
        dt = time.time() - t0
        ok = sum(match(f, data.get(f), film.get(f)) for f in FIELDS)
        console.print(f"[{i}/{len(films)}] {film['titel'][:32]:<32}  "
                      f"{de(len(cleaned)):>5} Zeichen  {de(dt):>4} s  {ok}/6 richtig")
        results.append({"gt": film, "data": data, "cleaned": cleaned, "sekunden": dt})

    print_results(results)
    avg = sum(r["sekunden"] for r in results) / len(results)
    console.print(f"im Schnitt {de(avg)} s pro Film (nur Modell, ohne Laden der Seite)")
    first = results[0]
    print_cast(first["gt"]["titel"], first["data"].get("besetzung"))

    if args.eingabe:
        lines = first["cleaned"].splitlines()
        shown = "\n".join(lines[:22]) + ("\n…" if len(lines) > 22 else "")
        console.print(Panel(shown, title=f"Das bekommt das Modell: {first['gt']['titel']} "
                                         f"({de(len(first['cleaned']))} Zeichen statt ca. 1,3 MB)",
                            border_style="blue"))
        console.print(Panel(json.dumps({k: first["data"].get(k) for k in FIELDS}, ensure_ascii=False, indent=2),
                            title="Das gibt das Modell zurück (JSON)", border_style="green"))

    if args.wiederholen:
        runs = [{k: first["data"].get(k) for k in FIELDS}]
        for _ in range(2):
            _, d = llm_extract((SNAP_DIR / "M0" / f"{first['gt']['imdb_id']}.html").read_text(encoding="utf-8"),
                               cleaner, client)
            runs.append({k: d.get(k) for k in FIELDS})
        same = all(r == runs[0] for r in runs)
        msg = ("[green]alle 3 Ausgaben sind identisch[/green]" if same
               else "[yellow]die Ausgaben unterscheiden sich[/yellow]")
        console.print(f"Wiederholung {first['gt']['titel']}, 3 Läufe: {msg}")

    if args.redesign:
        rt = Table(title="Redesign-Test M4: alle Klassen und data-testid umbenannt (richtige Felder von 6)")
        rt.add_column("Film")
        rt.add_column("XPath / CSS", justify="center")
        rt.add_column("LLM-Pipeline", justify="center")
        sx = sl = 0
        for r in results:
            film = r["gt"]
            html = (SNAP_DIR / "M4" / f"{film['imdb_id']}.html").read_text(encoding="utf-8")
            x = xpath_extract(html)
            _, d = llm_extract(html, cleaner, client)
            ox = sum(match(f, x.get(f), film.get(f)) for f in FIELDS)
            ol = sum(match(f, d.get(f), film.get(f)) for f in FIELDS)
            sx, sl = sx + ox, sl + ol
            rt.add_row(film["titel"], f"[red]{ox}/6[/red]" if ox < 6 else f"{ox}/6",
                       f"[green]{ol}/6[/green]" if ol >= 5 else f"{ol}/6")
        n = len(results) * len(FIELDS)
        rt.add_row("[bold]Summe[/bold]", f"[bold]{sx}/{n}[/bold]", f"[bold]{sl}/{n}[/bold]")
        console.print(rt)

    out = OUTPUT_DIR / f"demo_probelauf_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"metadata": {"erstellt": datetime.now().isoformat(), "llm_model": client.model,
                                "quelle": "robustness/mutations/M0 (gespeicherte IMDb-Seiten)"},
                   "movies": [{"imdb_id": r["gt"]["imdb_id"], "rang": r["gt"]["rang"], **r["data"]}
                              for r in results]},
                  fh, ensure_ascii=False, indent=2)
    console.print(f"\n[green]Ergebnis gespeichert:[/green] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
