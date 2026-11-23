"""
Evaluation der Besetzungs-Extraktion (m:n-Relation Film <-> Person).

Vergleicht die LLM-extrahierte Besetzung gegen den deterministischen
Ground Truth (ground_truth_cast.json) auf den 50 IMDb-Snapshots.

Metriken:
  * Schauspieler-F1: Mengen-Precision/Recall/F1 über die Schauspielernamen
    (fuzzy, case-/akzent-insensitiv). Pro Film gemittelt = Makro-F1.
  * Rollen-Genauigkeit: Anteil der korrekt erkannten Schauspieler, deren
    als_charakter ebenfalls mit dem GT übereinstimmt (Substring-Match).
  * 95%-Bootstrap-KI (B=10000) über die Filme.

Aufruf:
    python evaluation/evaluate_cast.py --model qwen2.5:3b [--n 50]
"""
import argparse
import json
import os
import re
import sys
import unicodedata
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scraper.cleaner import HTMLCleaner
from llm.ollama_client import OllamaClient

BASE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(os.path.dirname(BASE), "robustness", "snapshots")
GT_FILE = os.path.join(BASE, "ground_truth_cast.json")


def normalize(s: str) -> str:
    """Case- und akzent-insensitiv normalisieren für robusten Namensvergleich."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\(.*?\)", "", s)          # "(as Toshiro Mifune)" entfernen
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def role_match(pred: str, gt: str) -> bool:
    """Rollen gelten als gleich, wenn eine Substring der anderen ist (normalisiert)."""
    p, g = normalize(pred), normalize(gt)
    if not p or not g:
        return p == g
    return p in g or g in p


def eval_film(pred_cast, gt_cast):
    """Liefert (precision, recall, f1, n_role_correct, n_actor_correct) für einen Film."""
    gt_by_name = {normalize(e["name"]): e for e in gt_cast if e.get("name")}
    pred_by_name = {normalize(e["name"]): e for e in pred_cast if e.get("name")}

    gt_names = set(gt_by_name)
    pred_names = set(pred_by_name)
    tp = gt_names & pred_names

    precision = len(tp) / len(pred_names) if pred_names else (1.0 if not gt_names else 0.0)
    recall = len(tp) / len(gt_names) if gt_names else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # Rollen-Genauigkeit nur auf korrekt erkannten Schauspielern
    role_correct = 0
    for name in tp:
        if role_match(pred_by_name[name].get("als_charakter") or "",
                      gt_by_name[name].get("als_charakter") or ""):
            role_correct += 1
    return precision, recall, f1, role_correct, len(tp)


def bootstrap_ci(values, B=10000, seed=42):
    random.seed(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    means = []
    for _ in range(B):
        sample = [values[random.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (means[int(0.025 * B)], means[int(0.975 * B)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--n", type=int, default=50, help="Anzahl Filme (max 50)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt = {f["imdb_id"]: f["besetzung"] for f in json.load(open(GT_FILE))["filme"]}
    cleaner = HTMLCleaner()
    client = OllamaClient(model=args.model)

    ids = sorted(gt.keys())[: args.n]
    per_film_f1, per_film_prec, per_film_rec = [], [], []
    role_total, role_correct = 0, 0
    rows = []

    for i, imdb_id in enumerate(ids, 1):
        path = os.path.join(SNAPSHOT_DIR, imdb_id + ".html")
        html = open(path, encoding="utf-8", errors="ignore").read()
        cleaned = cleaner.clean_for_imdb(html)
        data = client.extract_film_data(cleaned, use_few_shot=True)
        pred_cast = (data or {}).get("besetzung", []) if data else []

        p, r, f1, rc, ntp = eval_film(pred_cast, gt[imdb_id])
        per_film_prec.append(p)
        per_film_rec.append(r)
        per_film_f1.append(f1)
        role_total += ntp
        role_correct += rc
        rows.append({
            "imdb_id": imdb_id, "precision": round(p, 3), "recall": round(r, 3),
            "f1": round(f1, 3), "n_pred": len(pred_cast), "n_gt": len(gt[imdb_id]),
            "role_correct": rc, "role_of": ntp,
        })
        print(f"[{i:2}/{len(ids)}] {imdb_id}: F1={f1:.2f} "
              f"(P={p:.2f} R={r:.2f}) Rollen {rc}/{ntp}")

    macro_f1 = sum(per_film_f1) / len(per_film_f1)
    macro_p = sum(per_film_prec) / len(per_film_prec)
    macro_r = sum(per_film_rec) / len(per_film_rec)
    ci = bootstrap_ci(per_film_f1)
    role_acc = role_correct / role_total if role_total else 0.0

    summary = {
        "model": args.model,
        "n_filme": len(ids),
        "schauspieler_precision": round(macro_p, 4),
        "schauspieler_recall": round(macro_r, 4),
        "schauspieler_f1": round(macro_f1, 4),
        "schauspieler_f1_ci95": [round(ci[0], 4), round(ci[1], 4)],
        "rollen_genauigkeit": round(role_acc, 4),
        "rollen_basis": role_total,
        "per_film": rows,
    }
    out = args.out or os.path.join(BASE, f"cast_eval_{args.model.replace(':', '_').replace('/', '_')}.json")
    json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"Modell: {args.model}  (n={len(ids)} Filme)")
    print(f"Schauspieler  P={macro_p:.3f}  R={macro_r:.3f}  F1={macro_f1:.3f}  "
          f"95%-KI [{ci[0]:.3f}; {ci[1]:.3f}]")
    print(f"Rollen-Genauigkeit: {role_acc:.3f}  ({role_correct}/{role_total} korrekt erkannte Schauspieler)")
    print(f"Geschrieben: {out}")


if __name__ == "__main__":
    main()
