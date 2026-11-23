"""
Lädt IMDb-Filmseiten als HTML-Snapshots zur Mutation.
Standard: 50 Filme (für die erweiterte Robustheits-Studie). Kann per CLI
mit --n überschrieben werden, um die ursprüngliche n=20-Konfiguration
zu reproduzieren. Bestehende Snapshots werden nicht erneut geladen.
"""
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from playwright.sync_api import sync_playwright

GT_PATH = Path(__file__).parent.parent / "evaluation" / "ground_truth_250.json"
OUT_DIR = Path(__file__).parent / "snapshots"
OUT_DIR.mkdir(exist_ok=True)

DEFAULT_N = 50
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    args = parser.parse_args()
    n = args.n

    with open(GT_PATH) as f:
        gt = json.load(f)
    films = gt["filme"][:n]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=USER_AGENT, locale="en-US")
        page = ctx.new_page()

        for i, film in enumerate(films, 1):
            imdb_id = film["imdb_id"]
            url = f"https://www.imdb.com/title/{imdb_id}/"
            target = OUT_DIR / f"{imdb_id}.html"
            if target.exists():
                print(f"[{i}/{n}] {imdb_id} ({film['titel']}) - cached")
                continue
            try:
                print(f"[{i}/{n}] {imdb_id} ({film['titel']})")
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_selector('h1, [data-testid="hero__pageTitle"]', timeout=10000)
                html = page.content()
                target.write_text(html, encoding="utf-8")
                time.sleep(2)
            except Exception as e:
                print(f"  Fehler: {e}")

        ctx.close()
        browser.close()
    print(f"\nSnapshots in {OUT_DIR} verfügbar.")


if __name__ == "__main__":
    main()
