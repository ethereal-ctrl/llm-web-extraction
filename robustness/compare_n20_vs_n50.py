"""
Vergleich der Robustheits-Studie n=20 (BA-Original) vs. n=50 (erweitert).

Liest robustness_results.json (alt, n=20) und robustness_results_n50.json
(neu) und stellt feldweise/Gesamt-F1 sowie ΔF1 inklusive CI gegenüber.
"""
import json
from pathlib import Path

OLD = Path(__file__).parent / "robustness_results.json"
NEW = Path(__file__).parent / "robustness_results_n50.json"

LEVELS = ["M0", "M1", "M2", "M3", "M4"]


def main():
    with open(OLD) as f:
        old = json.load(f)
    with open(NEW) as f:
        new = json.load(f)

    print(f"{'Stufe':<6} {'F1_LLM_n20':>12} {'F1_LLM_n50':>12} "
          f"{'F1_XP_n20':>12} {'F1_XP_n50':>12} "
          f"{'Δ_n20':>10} {'Δ_n50':>10} {'CI_n50':>22}")
    print("-" * 110)
    for lvl in LEVELS:
        l20 = old["llm"][lvl]["f1"]["GESAMT"]
        l50 = new["llm"][lvl]["f1"]["GESAMT"]
        x20 = old["xpath"][lvl]["f1"]["GESAMT"]
        x50 = new["xpath"][lvl]["f1"]["GESAMT"]
        d20 = l20 - x20
        d50 = new["delta"][lvl]["GESAMT"]
        ci = f"[{d50['ci_lo']:+.3f}; {d50['ci_hi']:+.3f}]"
        print(f"{lvl:<6} {l20:>12.3f} {l50:>12.3f} {x20:>12.3f} {x50:>12.3f} "
              f"{d20:>+10.3f} {d50['delta']:>+10.3f} {ci:>22}")

    print("\nFeldweiser ΔF1 (LLM − XPath) auf n=50 mit 95%-Bootstrap-CI:")
    print(f"{'Feld':<14} " + "".join(f"{l:>22}" for l in LEVELS))
    for f in ["titel", "jahr", "bewertung", "dauer_minuten", "regisseur", "genres"]:
        cells = []
        for lvl in LEVELS:
            d = new["delta"][lvl][f]
            cells.append(f"{d['delta']:+.2f}[{d['ci_lo']:+.2f};{d['ci_hi']:+.2f}]")
        print(f"{f:<14} " + "".join(f"{c:>22}" for c in cells))


if __name__ == "__main__":
    main()
