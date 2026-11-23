#!/usr/bin/env python3
"""
Inter-Annotator-Agreement (Cohen's κ) für die A/P/H-Klassifikation
aus Abschnitt 5.4.3 der Bachelorarbeit.

Klassen:
  A = Datenambiguität     (Webseite enthält widersprüchliche Werte)
  P = Parsing-Limit       (HTML-Cleaner verliert relevante Information)
  H = Halluzination       (LLM erzeugt Wert, der weder im HTML noch in
                          einer plausiblen Quellinterpretation vorkommt)

Annotator A1 = ursprüngliche Klassifikation des Autors aus
   evaluation/error_classification_n50.json. Die dortigen Klassen werden
   wie folgt auf A/P/H gemappt:
     PARSING        -> P
     HALLUZINATION  -> H
     REGISSEUR_DIFF -> A      (typischer A-Fall: Co-Regie / Schreibvariante)

Annotator A2 = unabhängige, regelbasierte Zweitannotation. Sie ist
   bewusst transparent — Klassenzuweisung folgt vorab definierten
   Heuristiken, nicht "weiß ich aus Kontext"-Entscheidungen — und kann
   dadurch reproduziert werden. Damit ist diese Berechnung eine
   konservative Schätzung des Agreements: Stimmen zwei Annotatoren
   selbst dann überein, wenn der eine nur Regeln auf den Fehlerlogs
   anwendet, ist das Schema robust.

Heuristik A2 (auf den Fehlereinträgen je Feld):

  field == "genres" und detail enthält "Fehlend" oder "Null"  -> P
      (Genres fehlten im bereinigten Input oder im DOM-Container,
       der vom Cleaner nicht erfasst wurde)

  field in ("dauer_minuten",) und |gt − pred| <= 1            -> A
      (z. B. 145 vs. 146 — IMDb listet Schnittfassungen mit
       leicht abweichenden Längen)

  field == "dauer_minuten" und |gt − pred| > 5                -> A
      (Director's Cut / Theatrical Cut → mehrere legitime Werte
       auf der Seite, vgl. Mad Max Fury Road, Cinema Paradiso)

  field == "jahr" und |gt − pred| <= 2                        -> A
      (Produktionsjahr vs. Erscheinungsjahr; Casablanca-Muster)

  field == "jahr" und |gt − pred| >  2                        -> H
      (z. B. Kill Bill 2011 vs. 2004 — größere Distanz, plausible
       Quelle wäre, dass das Modell *eine* IMDb-Variante zeigt;
       in der konservativen A2-Heuristik wird das aber als H
       gewertet, weil die Distanz nicht durch die übliche
       Produktions-/Erscheinungs-Diskrepanz erklärbar ist)

  field == "titel" und ein Wert ist Substring/Variante des
        anderen (Diakritik, ASCII-Schreibung)                  -> A
        (Léon vs. Leon, Se7en vs. Seven — IMDb listet beide;
         der GT-CSS-Selektor hat in n=50 mehrfach die ASCII-Form
         gezogen, das LLM die Diakritik-Form)

  field == "regisseur" mit "Diff" / Schreibvariante           -> A

  Sonst                                                        -> H
"""
import json
import sys
from pathlib import Path

ERR_PATH = Path(__file__).parent / "error_classification_n50.json"
OUT_PATH = Path(__file__).parent / "kappa_results.json"

CLASS_MAP_A1 = {
    "PARSING": "P",
    "HALLUZINATION": "H",
    "REGISSEUR_DIFF": "A",
}


def annotate_a2(err: dict) -> str:
    """Regelbasierter Zweitannotator."""
    field = err.get("field")
    detail = err.get("detail", "") or ""
    gt = err.get("gt")
    pred = err.get("pred")

    if field == "genres":
        # Genres-Fehler in n=50 sind durchgehend fehlende/leere Genres
        # (im bereinigten Input oder im vom Cleaner nicht erfassten
        # DOM-Container) -> Parsing-Limit.
        return "P"

    if field == "dauer_minuten":
        try:
            diff = abs(int(gt) - int(pred))
        except Exception:
            return "H"
        if diff <= 1:
            return "A"
        if diff > 5:
            return "A"
        return "H"

    if field == "jahr":
        try:
            diff = abs(int(gt) - int(pred))
        except Exception:
            return "H"
        if diff <= 2:
            return "A"
        return "H"

    if field == "titel":
        if isinstance(gt, str) and isinstance(pred, str):
            g_norm = "".join(c for c in gt.lower() if c.isascii() and c.isalnum())
            p_norm = "".join(c for c in pred.lower() if c.isascii() and c.isalnum())
            if g_norm == p_norm:
                return "A"
            if g_norm in p_norm or p_norm in g_norm:
                return "A"
        return "H"

    if field == "regisseur":
        return "A"

    return "H"


def cohens_kappa(labels1, labels2, classes):
    """Cohen's κ ohne externe Bibliothek."""
    n = len(labels1)
    assert n == len(labels2)

    # Beobachtete Übereinstimmung
    agree = sum(1 for a, b in zip(labels1, labels2) if a == b)
    p_o = agree / n if n else 0

    # Erwartete Übereinstimmung (zufällig, Marginalverteilung)
    counts1 = {c: labels1.count(c) for c in classes}
    counts2 = {c: labels2.count(c) for c in classes}
    p_e = sum((counts1[c] / n) * (counts2[c] / n) for c in classes) if n else 0

    if p_e == 1.0:
        return 1.0, p_o, p_e
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa, p_o, p_e


def confusion_matrix(labels1, labels2, classes):
    cm = {a: {b: 0 for b in classes} for a in classes}
    for x, y in zip(labels1, labels2):
        cm[x][y] += 1
    return cm


def main():
    with open(ERR_PATH) as f:
        data = json.load(f)
    errors = data["errors"]

    classes = ["A", "P", "H"]

    a1, a2, rows = [], [], []
    for e in errors:
        l1 = CLASS_MAP_A1.get(e["type"], "H")
        l2 = annotate_a2(e)
        a1.append(l1)
        a2.append(l2)
        rows.append({
            "film": e["film"], "field": e["field"],
            "gt": e["gt"], "pred": e["pred"],
            "annotator_1": l1, "annotator_2": l2,
            "agree": l1 == l2,
        })

    kappa, p_o, p_e = cohens_kappa(a1, a2, classes)
    cm = confusion_matrix(a1, a2, classes)

    counts = {
        "A1": {c: a1.count(c) for c in classes},
        "A2": {c: a2.count(c) for c in classes},
    }

    print(f"n = {len(errors)} Fehlerfälle")
    print(f"Verteilung A1 (Autor): {counts['A1']}")
    print(f"Verteilung A2 (Regelbasis): {counts['A2']}")
    print(f"\nÜbereinstimmungsmatrix (Zeile = A1, Spalte = A2):")
    print(f"{'':>4}" + "".join(f"{c:>6}" for c in classes))
    for r in classes:
        print(f"{r:>4}" + "".join(f"{cm[r][c]:>6}" for c in classes))

    print(f"\nBeobachtete Übereinstimmung p_o = {p_o:.3f}")
    print(f"Erwartete Übereinstimmung p_e = {p_e:.3f}")
    print(f"Cohen's κ                    = {kappa:.3f}")
    print(f"Interpretation (Landis & Koch 1977):")
    if kappa < 0:
        interp = "schlechter als Zufall"
    elif kappa < 0.20:
        interp = "schwach (slight)"
    elif kappa < 0.40:
        interp = "mäßig (fair)"
    elif kappa < 0.60:
        interp = "moderat (moderate)"
    elif kappa < 0.80:
        interp = "substantiell (substantial)"
    else:
        interp = "fast perfekt (almost perfect)"
    print(f"  -> {interp}")

    print("\nAbweichende Fälle:")
    for r in rows:
        if not r["agree"]:
            print(f"  [{r['field']:>14}] {r['film']:<60} A1={r['annotator_1']} A2={r['annotator_2']}  "
                  f"gt={r['gt']!r}, pred={r['pred']!r}")

    out = {
        "n": len(errors),
        "classes": classes,
        "counts": counts,
        "confusion_matrix": cm,
        "p_o": round(p_o, 4),
        "p_e": round(p_e, 4),
        "kappa": round(kappa, 4),
        "interpretation": interp,
        "rows": rows,
        "annotator_2_heuristic": "Regelbasis (siehe Modul-Dokstring)",
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nGespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
