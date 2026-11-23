#!/usr/bin/env python3
"""
Vergleichsstudie: LLM-Direktextraktion auf Güngörs Testdomänen.

Vergleicht direkten LLM-Extraktionsansatz (dieser Arbeit) mit
Güngörs XPath-Generierungsansatz auf identischen Testdomänen.

Testdomänen:
  1. quotes.toscrape.com  (10 Seiten × 10 Items = 100 Items, 3 Felder)
  2. books.toscrape.com   (3 Seiten × 20 Items = 60 Items, 4 Felder)

Verwendung:
    python toscrape_compare.py                        # beide Sites, gemma3:4b
    python toscrape_compare.py --site quotes          # nur Quotes
    python toscrape_compare.py --site books           # nur Books
    python toscrape_compare.py --max-items 20         # Test mit 20 Items
    OLLAMA_MODEL=qwen2.5:3b python toscrape_compare.py
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sites.quotes_toscrape import QUOTES_CONFIG
from sites.books_toscrape import BOOKS_LIST_CONFIG
from scraper.generic_scraper import GenericScraper
from llm.generic_extractor import GenericExtractor
from evaluation.evaluate_site import evaluate_site
from config import OLLAMA_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_site(config, model: str, max_items=None, output_dir: Path = None):
    """
    Führt Scraping + LLM-Extraktion für eine Site durch.

    Returns:
        (eval_results, output_path)
    """
    output_dir = output_dir or Path("output")
    output_dir.mkdir(exist_ok=True)

    site_slug = (
        config.name
        .replace(" ", "_")
        .replace(".", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"toscrape_{site_slug}_{timestamp}.json"

    print(f"\n{'='*60}")
    print(f"  Site:   {config.name}")
    print(f"  Modell: {model}")
    if max_items:
        print(f"  Limit:  {max_items} Items")
    print(f"{'='*60}")

    # --- Schritt 1: Scraping ---
    t_scrape_start = time.time()
    scraper = GenericScraper(config)
    items = scraper.scrape_all(max_items=max_items)
    t_scrape = time.time() - t_scrape_start
    logger.info(f"Scraping abgeschlossen: {len(items)} Items in {t_scrape:.1f}s")

    # --- Schritt 2: LLM-Extraktion ---
    t_extract_start = time.time()
    extractor = GenericExtractor(config, model=model)
    results = extractor.extract_batch(items)
    t_extract = time.time() - t_extract_start

    # Token-Statistiken
    total_prompt_tokens = sum(
        r.get("_meta", {}).get("prompt_tokens") or 0 for r in results
    )
    total_response_tokens = sum(
        r.get("_meta", {}).get("response_tokens") or 0 for r in results
    )
    avg_elapsed = (
        sum(r.get("_meta", {}).get("elapsed_s", 0) for r in results) / len(results)
        if results else 0
    )

    logger.info(
        f"Extraktion abgeschlossen: {len(results)} Items in {t_extract:.1f}s "
        f"(Ø {avg_elapsed:.1f}s/Item)"
    )
    logger.info(
        f"Tokens: {total_prompt_tokens} Prompt + {total_response_tokens} Response"
    )

    # --- Schritt 3: Speichern ---
    output = {
        "site": config.name,
        "model": model,
        "timestamp": timestamp,
        "n_items": len(results),
        "timing": {
            "scrape_s": round(t_scrape, 2),
            "extract_s": round(t_extract, 2),
            "total_s": round(t_scrape + t_extract, 2),
            "avg_per_item_s": round(avg_elapsed, 2),
        },
        "tokens": {
            "total_prompt": total_prompt_tokens,
            "total_response": total_response_tokens,
        },
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Output gespeichert: {output_path}")

    # --- Schritt 4: Evaluation ---
    eval_results = evaluate_site(config, str(output_path))

    # Timing-Info zur Evaluation hinzufügen
    eval_results["timing"] = output["timing"]
    eval_results["tokens"] = output["tokens"]

    return eval_results, str(output_path)


def print_summary(all_results: dict):
    """Druckt Zusammenfassung aller Sites."""
    print("\n" + "=" * 60)
    print("  ZUSAMMENFASSUNG")
    print("=" * 60)
    print(f"  {'Site':<20} {'F1':>6}  {'Coverage':>9}  {'Zeit':>8}  {'Tokens/Item':>12}")
    print(f"  {'-'*20} {'-'*6}  {'-'*9}  {'-'*8}  {'-'*12}")

    for site_name, res in all_results.items():
        f1 = res["gesamt"]["f1"]
        cov = res["coverage"]
        timing = res.get("timing", {})
        tokens = res.get("tokens", {})
        total_s = timing.get("total_s", 0)
        n = res["n_items"]
        tokens_per_item = (
            (tokens.get("total_prompt", 0) + tokens.get("total_response", 0)) / n
            if n else 0
        )
        print(
            f"  {site_name:<20} {f1:>6.3f}  {cov:>9.0%}  "
            f"{total_s:>7.1f}s  {tokens_per_item:>11.0f}"
        )

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Vergleichsstudie: Direktextraktion auf Güngörs Testdomänen"
    )
    parser.add_argument(
        "--site", choices=["quotes", "books", "all"], default="all",
        help="Welche Site(s) laufen lassen (Standard: all)"
    )
    parser.add_argument(
        "--max-items", type=int, default=None,
        help="Maximale Anzahl Items (für Tests)"
    )
    parser.add_argument(
        "--model", default=None,
        help="Ollama-Modell (Standard: aus config.py oder $OLLAMA_MODEL)"
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Ausgabeverzeichnis (Standard: output/)"
    )
    args = parser.parse_args()

    model = args.model or os.environ.get("OLLAMA_MODEL") or OLLAMA_CONFIG["model"]
    output_dir = Path(args.output_dir)

    print(f"\nVergleichsstudie: LLM-Direktextraktion auf toscrape.com")
    print(f"Modell: {model}")
    print(f"Datum:  {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    all_results = {}

    if args.site in ("quotes", "all"):
        result, path = run_site(
            QUOTES_CONFIG, model=model,
            max_items=args.max_items, output_dir=output_dir
        )
        all_results["quotes.toscrape.com"] = result

    if args.site in ("books", "all"):
        result, path = run_site(
            BOOKS_LIST_CONFIG, model=model,
            max_items=args.max_items, output_dir=output_dir
        )
        all_results["books.toscrape.com"] = result

    if len(all_results) > 1:
        print_summary(all_results)


if __name__ == "__main__":
    main()
