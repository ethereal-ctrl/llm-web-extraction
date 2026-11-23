"""
Erzeugt mutierte HTML-Varianten der gespeicherten Snapshots.

Mutationen simulieren typische Layout-Änderungen, wie sie bei IMDb-Redesigns
auftreten: Umbenennung von CSS-Klassen, Vertauschung/Entfernung von
data-testid-Attributen, leichte DOM-Verschachtelung.

Mutationsstärken:
  M0: keine Mutation (Baseline)
  M1: leichte Mutation - 30% der CSS-Klassen umbenannt, 0% data-testid
  M2: mittlere Mutation - 60% CSS-Klassen, 30% data-testid umbenannt
  M3: starke Mutation - 100% CSS-Klassen, 60% data-testid umbenannt
  M4: aggressive Mutation - 100% CSS-Klassen, 100% data-testid umbenannt
"""
import re
import random
from pathlib import Path
from bs4 import BeautifulSoup

IN_DIR = Path(__file__).parent / "snapshots"
OUT_DIR = Path(__file__).parent / "mutations"
OUT_DIR.mkdir(exist_ok=True)

# Reproduzierbare Mutation
SEED = 42

# Mutationsstufen (Anteil class-Umbenennung, Anteil data-testid-Umbenennung)
LEVELS = {
    "M0": (0.0, 0.0),
    "M1": (0.3, 0.0),
    "M2": (0.6, 0.3),
    "M3": (1.0, 0.6),
    "M4": (1.0, 1.0),
}


def mutate_html(html: str, p_class: float, p_testid: float, rng: random.Random) -> str:
    """Mutiert HTML reproduzierbar."""
    soup = BeautifulSoup(html, "lxml")
    # Klassen umbenennen
    for tag in soup.find_all(class_=True):
        classes = tag.get("class") or []
        new_classes = []
        for c in classes:
            if rng.random() < p_class:
                # In randomisierte Variante umbenennen, gleiche Länge ungefähr
                new_classes.append(f"x-{rng.randint(10000, 99999)}")
            else:
                new_classes.append(c)
        tag["class"] = new_classes
    # data-testid umbenennen
    for tag in soup.find_all(attrs={"data-testid": True}):
        if rng.random() < p_testid:
            tag["data-testid"] = f"renamed-{rng.randint(10000, 99999)}"
    return str(soup)


def main():
    snapshots = sorted(IN_DIR.glob("*.html"))
    print(f"Mutiere {len(snapshots)} Snapshots auf {len(LEVELS)} Stufen "
          f"= {len(snapshots) * len(LEVELS)} Dateien")

    for snap in snapshots:
        html = snap.read_text(encoding="utf-8")
        for level, (pc, pt) in LEVELS.items():
            rng = random.Random(SEED + hash(level + snap.stem) % 10000)
            target_dir = OUT_DIR / level
            target_dir.mkdir(exist_ok=True)
            target = target_dir / snap.name
            mutated = mutate_html(html, pc, pt, rng)
            target.write_text(mutated, encoding="utf-8")
        print(f"  {snap.stem}: 5 Stufen erzeugt")

    print(f"\nFertig: {OUT_DIR}/")


if __name__ == "__main__":
    main()
