"""
Ground-Truth-Generator für die Besetzung (m:n-Relation Film <-> Person).

Zieht die Besetzung deterministisch (ohne LLM) aus den 50 HTML-Snapshots
in robustness/snapshots/ über dieselben data-testid-Selektoren, die auch
der Cleaner nutzt (HTMLCleaner.extract_cast). Ausgabe: ground_truth_cast.json

Analog zum skalaren Ground Truth (generate_ground_truth_250.py) sichert die
deterministische Erzeugung Reproduzierbarkeit und erlaubt Bootstrap-KIs.
"""
import json
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scraper.cleaner import HTMLCleaner

SNAPSHOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "robustness", "snapshots",
)
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth_cast.json")

# Wie viele Cast-Einträge je Film in den GT aufgenommen werden. IMDb zeigt
# initial 18; wir werten die ersten 10 (Hauptbesetzung), weil tiefere Ränge
# zwischen Snapshots und Re-Renders am stärksten schwanken.
GT_CAST_DEPTH = 10


def main():
    cleaner = HTMLCleaner()
    files = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "*.html")))
    filme = []
    total_actors = 0
    for f in files:
        imdb_id = os.path.basename(f).replace(".html", "")
        html = open(f, encoding="utf-8", errors="ignore").read()
        cast = cleaner.extract_cast(html, max_cast=GT_CAST_DEPTH)
        total_actors += len(cast)
        filme.append({"imdb_id": imdb_id, "besetzung": cast})

    out = {
        "metadata": {
            "description": "Deterministische Ground-Truth für die Besetzung "
                           "(m:n Film<->Person) via data-testid-Selektoren",
            "quelle": "robustness/snapshots/*.html (50 IMDb-Filme)",
            "cast_tiefe": GT_CAST_DEPTH,
            "n_filme": len(filme),
            "n_actor_eintraege": total_actors,
        },
        "filme": filme,
    }
    with open(OUTPUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print(f"Ground Truth Cast geschrieben: {OUTPUT}")
    print(f"  Filme: {len(filme)}")
    print(f"  Cast-Einträge gesamt: {total_actors} "
          f"(Ø {total_actors / len(filme):.1f} je Film)")
    # Beispielausgabe
    ex = filme[0]
    print(f"  Beispiel {ex['imdb_id']}: "
          f"{', '.join(e['name'] + '→' + str(e['als_charakter']) for e in ex['besetzung'][:3])}")


if __name__ == "__main__":
    main()
