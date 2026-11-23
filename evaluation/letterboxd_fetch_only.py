"""Holt nur Letterboxd-Snapshots und erzeugt GT (kein LLM-Aufruf)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.letterboxd_runner import (
    fetch_snapshots, parse_ground_truth, SLUGS, OUT_DIR, GT_PATH, FIELDS,
)
import json


def main():
    paths = fetch_snapshots(SLUGS)
    print(f"\nSnapshots: {len(paths)}")

    gt = {}
    for p in paths:
        html = p.read_text(encoding="utf-8")
        gt[p.stem] = parse_ground_truth(html)
    GT_PATH.write_text(json.dumps(gt, indent=2, ensure_ascii=False))

    n_complete = sum(
        1 for v in gt.values() if v["titel"] and v["jahr"] and v["regisseur"]
    )
    print(f"GT komplett: {n_complete}/{len(gt)}")

    # Lücken zeigen
    for slug, v in gt.items():
        missing = [f for f in FIELDS if v.get(f) in (None, [])]
        if missing:
            print(f"  {slug:<48} fehlt: {missing}")


if __name__ == "__main__":
    main()
