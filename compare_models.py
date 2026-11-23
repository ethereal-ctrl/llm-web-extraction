#!/usr/bin/env python3
"""
Modellvergleich: qwen3:8b vs qwen2.5:3b
Vergleicht zwei JSON-Ausgaben des Scrapers field-by-field.
"""
import json
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

FELDER = ["titel", "jahr", "bewertung", "anzahl_bewertungen",
          "dauer_minuten", "regisseur", "genres", "beschreibung"]


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Index nach imdb_id
    return {m["imdb_id"]: m for m in data["movies"] if m.get("imdb_id")}


def compare(path_a: str, path_b: str, label_a: str, label_b: str):
    films_a = load_json(path_a)
    films_b = load_json(path_b)

    gemeinsame_ids = sorted(set(films_a) & set(films_b),
                            key=lambda x: films_a[x].get("rang", 999))

    console.print(f"\n[bold cyan]Modellvergleich: {label_a} vs {label_b}[/bold cyan]")
    console.print(f"Gemeinsame Filme: {len(gemeinsame_ids)} "
                  f"(nur in {label_a}: {len(set(films_a)-set(films_b))}, "
                  f"nur in {label_b}: {len(set(films_b)-set(films_a))})\n")

    abweichungen_gesamt = 0

    for imdb_id in gemeinsame_ids:
        film_a = films_a[imdb_id]
        film_b = films_b[imdb_id]
        rang = film_a.get("rang", "?")
        titel = film_a.get("titel", imdb_id)

        abweichungen = []
        for feld in FELDER:
            val_a = film_a.get(feld)
            val_b = film_b.get(feld)

            if feld == "genres":
                set_a = set(val_a or [])
                set_b = set(val_b or [])
                if set_a != set_b:
                    abweichungen.append((feld, str(sorted(set_a)), str(sorted(set_b))))
            elif feld == "beschreibung":
                # Nur prüfen ob beide vorhanden/leer
                both_present = bool(val_a) == bool(val_b)
                if not both_present:
                    abweichungen.append((feld,
                                         f"{len(val_a or '')} Zeichen",
                                         f"{len(val_b or '')} Zeichen"))
            elif val_a != val_b:
                abweichungen.append((feld, str(val_a), str(val_b)))

        status = "[green]✓ identisch[/green]" if not abweichungen else f"[yellow]{len(abweichungen)} Abweichung(en)[/yellow]"
        console.print(f"[bold]#{rang} {titel}[/bold] — {status}")

        if abweichungen:
            table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
            table.add_column("Feld", style="cyan", width=20)
            table.add_column(label_a, width=35)
            table.add_column(label_b, width=35)
            for feld, va, vb in abweichungen:
                table.add_row(feld, va or "[dim]None[/dim]", vb or "[dim]None[/dim]")
            console.print(table)
            abweichungen_gesamt += len(abweichungen)

    # Zusammenfassung
    console.print("\n" + "─" * 70)
    table = Table(title="Zusammenfassung", box=box.ROUNDED)
    table.add_column("Metrik", style="cyan")
    table.add_column("Wert", style="green")
    table.add_row("Verglichene Filme", str(len(gemeinsame_ids)))
    table.add_row("Gesamte Feldabweichungen", str(abweichungen_gesamt))
    if gemeinsame_ids:
        abw_rate = abweichungen_gesamt / (len(gemeinsame_ids) * len(FELDER)) * 100
        table.add_row("Abweichungsrate", f"{abw_rate:.1f}%")
    console.print(table)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        console.print("[red]Verwendung: python compare_models.py <json_a> <label_a> <json_b> <label_b>[/red]")
        console.print("Beispiel: python compare_models.py output/qwen3_10.json qwen3:8b output/qwen25_10.json qwen2.5:3b")
        sys.exit(1)

    compare(sys.argv[1], sys.argv[3], sys.argv[2], sys.argv[4])
