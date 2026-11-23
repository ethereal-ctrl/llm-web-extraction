#!/usr/bin/env python3
"""
GT-Audit-Tool: Manuelle Verifikation der Ground-Truth-Datei
Zieht n=50 zufällige Filme, öffnet die IMDb-Seite und zeigt GT-Werte zum Abhaken.
Am Ende: Wilson-KI + LaTeX-Textbaustein für Threats-Sektion.

Verwendung: python3 gt_audit.py [--seed 42] [--n 50] [--resume]
"""

import json
import random
import math
import argparse
import webbrowser
import os
import sys
from datetime import datetime

AUDIT_STATE_FILE = "gt_audit_state.json"

def wilson_ci(k, n, z=1.96):
    """Wilson-Konfidenzintervall für Proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)

def load_gt():
    path = os.path.join(os.path.dirname(__file__), "ground_truth_250.json")
    with open(path) as f:
        data = json.load(f)
    return data["filme"]

def save_state(state):
    with open(os.path.join(os.path.dirname(__file__), AUDIT_STATE_FILE), "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def load_state():
    path = os.path.join(os.path.dirname(__file__), AUDIT_STATE_FILE)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def format_genres(genres):
    if isinstance(genres, list):
        return ", ".join(genres)
    return str(genres)

def print_film(i, total, film):
    print("\n" + "="*60)
    print(f"  Film {i}/{total}  |  Rang #{film['rang']}  |  {film['imdb_id']}")
    print("="*60)
    print(f"  Titel:      {film['titel']}")
    print(f"  Jahr:       {film['jahr']}")
    print(f"  Bewertung:  {film['bewertung']}")
    print(f"  Laufzeit:   {film['dauer_minuten']} min")
    print(f"  Regisseur:  {film['regisseur']}")
    print(f"  Genres:     {format_genres(film.get('genres', 'N/A'))}")
    print()

def ask_field(field_name, gt_value, errors):
    """Fragt ob ein Feld korrekt ist. Gibt zurück: True=korrekt, False=Fehler, None=drift/unklar."""
    while True:
        inp = input(f"  [{field_name}] GT='{gt_value}'  Korrekt? [y/n/d=drift/s=skip] ").strip().lower()
        if inp in ("y", ""):
            return True
        elif inp == "n":
            detail = input(f"    Tatsächlicher Wert auf IMDb: ").strip()
            errors.append({"feld": field_name, "gt": str(gt_value), "imdb": detail})
            return False
        elif inp == "d":
            detail = input(f"    Abweichung (z.B. Bewertung jetzt 9.2): ").strip()
            errors.append({"feld": field_name, "gt": str(gt_value), "imdb": detail, "typ": "drift"})
            return None  # drift: zählt nicht als GT-Fehler
        elif inp == "s":
            return None  # übersprungen

def audit_film(i, total, film):
    print_film(i, total, film)
    url = f"https://www.imdb.com/title/{film['imdb_id']}/"
    open_browser = input(f"  IMDb öffnen? [{url}] [y/n] ").strip().lower()
    if open_browser != "n":
        webbrowser.open(url)
        input("  (Seite geladen? Enter zum Weitermachen) ")

    errors = []
    results = {}
    fields = [
        ("titel",          film["titel"]),
        ("jahr",           film["jahr"]),
        ("bewertung",      film["bewertung"]),
        ("dauer_minuten",  film["dauer_minuten"]),
        ("regisseur",      film["regisseur"]),
        ("genres",         format_genres(film.get("genres", "N/A"))),
    ]
    for fname, fval in fields:
        ok = ask_field(fname, fval, errors)
        results[fname] = ok  # True/False/None

    return results, errors

def compute_stats(all_results):
    """Berechnet Gesamt-Accuracy über alle Felder, exklusive drift/skip."""
    total = 0
    correct = 0
    for film_id, fields in all_results.items():
        for fname, val in fields.items():
            if val is True:
                total += 1
                correct += 1
            elif val is False:
                total += 1
            # None (drift/skip) ignorieren
    return correct, total

def print_report(all_results, all_errors, sample_ids, seed, n):
    correct, total = compute_stats(all_results)
    lo, hi = wilson_ci(correct, total)

    print("\n" + "="*60)
    print("  AUDIT-ERGEBNIS")
    print("="*60)
    print(f"  Stichprobe:    n={n} Filme (Seed={seed}), {total} Felder bewertet")
    print(f"  Korrekt:       {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"  Wilson-95%-KI: [{lo:.3f}; {hi:.3f}]")

    if all_errors:
        print(f"\n  Fehler/Drift ({len(all_errors)}):")
        for e in all_errors:
            typ = e.get("typ", "fehler")
            print(f"    [{typ.upper()}] {e['feld']}: GT='{e['gt']}' → IMDb='{e['imdb']}'")

    print("\n--- LaTeX-Textbaustein (Threats to Validity) ---\n")
    latex = (
        f"Die manuelle Stichprobenprüfung umfasst $n={n}$~Filme "
        f"(zufällig gezogen, Seed={seed}), "
        f"für die alle sechs Felder (Titel, Jahr, Bewertung, Laufzeit, Regisseur, Genres) "
        f"händisch gegen die IMDb-Seite abgeglichen wurden. "
        f"Von insgesamt {total}~bewerteten Feldern "
        f"stimmten {correct}~({100*correct/total:.1f}\\,\\%) mit dem Ground-Truth überein. "
        f"Das zugehörige Wilson-95\\,\\%-Konfidenzintervall beträgt "
        f"$[{lo:.2f};\\;{hi:.2f}]$. "
    )
    if correct == total:
        latex += (
            f"Alle geprüften Felder waren korrekt; die untere Intervallgrenze von "
            f"{lo:.2f} liegt damit oberhalb des gemessenen Gesamt-F1 von 0{{,}}97 und "
            f"stützt die Validität der Ground Truth."
        )
    else:
        latex += (
            f"Die {total - correct}~abweichenden Felder sind im Audit-Protokoll "
            f"dokumentiert und als Einschränkung bei der Interpretation der F1-Werte "
            f"zu berücksichtigen."
        )
    print(latex)
    print()
    return {"correct": correct, "total": total, "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4)}

def main():
    parser = argparse.ArgumentParser(description="GT-Audit für IMDb-Top-250-Ground-Truth")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n", type=int, default=50, help="Stichprobengröße (default: 50)")
    parser.add_argument("--resume", action="store_true", help="Unterbrochenes Audit fortsetzen")
    args = parser.parse_args()

    filme = load_gt()

    # Stichprobe ziehen
    rng = random.Random(args.seed)
    sample = rng.sample(filme, args.n)

    # Resume-Logik
    state = None
    start_idx = 0
    all_results = {}
    all_errors = []

    if args.resume:
        state = load_state()
        if state and state.get("seed") == args.seed and state.get("n") == args.n:
            all_results = state.get("results", {})
            all_errors = state.get("errors", [])
            start_idx = state.get("next_idx", 0)
            print(f"\n  Fortsetze Audit ab Film {start_idx+1}/{args.n}...")
        else:
            print("  Kein passender Audit-Stand gefunden. Starte neu.")

    print(f"\n  GT-Audit: {args.n} zufällige Filme, Seed={args.seed}")
    print(f"  Prüfdatum: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"  Felder: titel, jahr, bewertung, dauer_minuten, regisseur, genres")
    print(f"\n  Steuerung: y=korrekt  n=falsch  d=drift(Bewertungsänderung etc.)  s=skip")
    print(f"  Jederzeit Ctrl+C → Fortschritt wird gespeichert.\n")

    try:
        for i, film in enumerate(sample):
            if i < start_idx:
                continue
            results, errors = audit_film(i + 1, args.n, film)
            all_results[film["imdb_id"]] = results
            all_errors.extend(errors)

            # Zwischenstand speichern
            save_state({
                "seed": args.seed,
                "n": args.n,
                "next_idx": i + 1,
                "results": all_results,
                "errors": all_errors,
                "date": datetime.now().strftime("%Y-%m-%d"),
            })

    except KeyboardInterrupt:
        print("\n\n  Unterbrochen. Fortschritt gespeichert.")
        print(f"  Weiter mit: python3 gt_audit.py --seed {args.seed} --n {args.n} --resume")
        sys.exit(0)

    # Abschluss-Report
    stats = print_report(all_results, all_errors, sample, args.seed, args.n)

    # Finalen Stand speichern
    save_state({
        "seed": args.seed,
        "n": args.n,
        "next_idx": args.n,
        "results": all_results,
        "errors": all_errors,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "stats": stats,
        "complete": True,
    })
    print(f"  Audit abgeschlossen. Stand gespeichert in: {AUDIT_STATE_FILE}")

if __name__ == "__main__":
    main()
