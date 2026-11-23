"""
Deterministische Ground-Truth-Generierung für IMDb Top 250.
Verwendet ausschließlich CSS-Selektoren und Regex – kein LLM.
"""
import json
import re
import time
import logging
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Offizielle IMDb-Genres (für Validierung)
VALID_GENRES = {
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "History",
    "Horror", "Music", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Short", "Sport", "Thriller", "War", "Western",
}

BASE_URL = "https://www.imdb.com"
TOP_250_URL = "https://www.imdb.com/chart/top/"


def extract_film_data(html: str, rang: int) -> dict:
    """Extrahiert Filmdaten deterministisch aus HTML via CSS-Selektoren."""
    soup = BeautifulSoup(html, "lxml")
    data = {"rang": rang}

    # IMDb ID aus canonical URL
    canonical = soup.select_one('link[rel="canonical"]')
    if canonical:
        m = re.search(r'/title/(tt\d+)', canonical.get("href", ""))
        if m:
            data["imdb_id"] = m.group(1)

    # Titel
    title_elem = soup.select_one('[data-testid="hero__pageTitle"]')
    if title_elem:
        data["titel"] = title_elem.get_text(strip=True)

    # Bewertung
    rating_elem = soup.select_one('[data-testid="hero-rating-bar__aggregate-rating__score"] span')
    if rating_elem:
        try:
            data["bewertung"] = float(rating_elem.get_text(strip=True))
        except ValueError:
            pass

    # Jahr aus Release-Datum
    release = soup.select_one('[data-testid="title-details-releasedate"] .ipc-metadata-list-item__list-content-item')
    if release:
        year_match = re.search(r'\b(19|20)\d{2}\b', release.get_text())
        if year_match:
            data["jahr"] = int(year_match.group(0))

    # Fallback Jahr: aus Metadaten-Block (älteres Layout)
    if "jahr" not in data:
        metadata = soup.select_one('[data-testid="hero-title-block__metadata"]')
        if metadata:
            year_match = re.search(r'\b(19|20)\d{2}\b', metadata.get_text())
            if year_match:
                data["jahr"] = int(year_match.group(0))

    # Laufzeit aus techspec
    runtime_elem = soup.select_one('[data-testid="title-techspec_runtime"] .ipc-metadata-list-item__content-container')
    if runtime_elem:
        rt_text = runtime_elem.get_text(strip=True)
        # Format: "2h 45m(165 min)" oder "1h 36m"
        rt_match = re.search(r'(\d+)\s*h\s*(\d+)\s*m', rt_text)
        if rt_match:
            data["dauer_minuten"] = int(rt_match.group(1)) * 60 + int(rt_match.group(2))
        else:
            h_match = re.search(r'(\d+)\s*h', rt_text)
            m_match = re.search(r'(\d+)\s*m', rt_text)
            total = 0
            if h_match:
                total += int(h_match.group(1)) * 60
            if m_match:
                total += int(m_match.group(1))
            if total > 0:
                data["dauer_minuten"] = total

    # Fallback Laufzeit: Metadaten-Block
    if "dauer_minuten" not in data:
        metadata = soup.select_one('[data-testid="hero-title-block__metadata"]')
        if metadata:
            rt_match = re.search(r'(\d+)\s*h\s*(\d+)\s*m', metadata.get_text())
            if rt_match:
                data["dauer_minuten"] = int(rt_match.group(1)) * 60 + int(rt_match.group(2))

    # Genres: aus .ipc-chip__text, gefiltert auf valide IMDb-Genres
    chip_elems = soup.select('.ipc-chip__text')
    if chip_elems:
        all_chips = [c.get_text(strip=True) for c in chip_elems]
        valid = [g for g in all_chips if g in VALID_GENRES]
        if valid:
            data["genres"] = valid

    # Fallback Genres: data-testid="genres"
    if "genres" not in data:
        genre_elems = soup.select('[data-testid="genres"] a .ipc-chip__text')
        if genre_elems:
            data["genres"] = [g.get_text(strip=True) for g in genre_elems]

    # Regisseur
    credit_sections = soup.select('[data-testid="title-pc-principal-credit"]')
    for section in credit_sections:
        label = section.select_one('.ipc-metadata-list-item__label')
        if label and ('Director' in label.get_text() or 'Regisseur' in label.get_text()):
            director_links = section.select('a')
            directors = [a.get_text(strip=True) for a in director_links
                        if a.get_text(strip=True) and a.get_text(strip=True) != label.get_text(strip=True)]
            if directors:
                data["regisseur"] = directors[0]
            break

    # Fallback Regisseur: erster principal-credit Link
    if "regisseur" not in data:
        first_credit = soup.select_one('[data-testid="title-pc-principal-credit"] a')
        if first_credit:
            text = first_credit.get_text(strip=True)
            if text and text not in ("Director", "Directors", "Regisseur"):
                data["regisseur"] = text

    return data


def main():
    existing_gt_path = Path(__file__).parent / "ground_truth.json"
    output_path = Path(__file__).parent / "ground_truth_250.json"

    # Lade existierende GT als Referenz
    existing = {}
    if existing_gt_path.exists():
        with open(existing_gt_path) as f:
            gt_data = json.load(f)
            for film in gt_data["filme"]:
                existing[film["imdb_id"]] = film

    logger.info(f"Existierende GT: {len(existing)} Filme")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            locale="en-US",
        )
        page = context.new_page()

        # Top 250 Liste laden
        logger.info("Lade Top 250 Liste...")
        page.goto(TOP_250_URL, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_selector('li.ipc-metadata-list-summary-item', timeout=10000)
        time.sleep(2)

        html = page.content()
        soup = BeautifulSoup(html, "lxml")
        movie_items = soup.select('li.ipc-metadata-list-summary-item')
        logger.info(f"{len(movie_items)} Filme in Liste gefunden")

        # Film-URLs sammeln
        film_urls = []
        for idx, item in enumerate(movie_items, 1):
            link = item.select_one('a[href*="/title/"]')
            if not link:
                continue
            href = link.get("href", "")
            url = BASE_URL + href.split("?")[0]
            imdb_id_match = re.search(r'/title/(tt\d+)', href)
            imdb_id = imdb_id_match.group(1) if imdb_id_match else None
            titel = re.sub(r'^\d+\.\s*', '', link.get_text(strip=True))
            film_urls.append({"rang": idx, "url": url, "imdb_id": imdb_id, "titel": titel})

        logger.info(f"{len(film_urls)} Film-URLs extrahiert")

        # Jeden Film scrapen
        filme = []
        errors = []

        for info in film_urls:
            rang = info["rang"]
            url = info["url"]
            imdb_id = info["imdb_id"]

            # Prüfe ob schon in existierender GT
            if imdb_id in existing:
                filme.append(existing[imdb_id])
                logger.info(f"[{rang}/250] {info['titel']} — aus existierender GT")
                continue

            try:
                logger.info(f"[{rang}/250] Scrape: {info['titel']}")
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_selector('[data-testid="hero__pageTitle"], h1', timeout=10000)
                time.sleep(1.5)

                film_html = page.content()
                data = extract_film_data(film_html, rang)

                # IMDb ID sicherstellen
                if "imdb_id" not in data:
                    data["imdb_id"] = imdb_id

                # Validierung
                required = ["titel", "jahr", "bewertung", "dauer_minuten", "regisseur", "genres"]
                missing = [f for f in required if f not in data]
                if missing:
                    logger.warning(f"  Fehlende Felder: {missing}")
                    errors.append({"rang": rang, "titel": info["titel"], "missing": missing})

                filme.append(data)

            except Exception as e:
                logger.error(f"  FEHLER: {e}")
                errors.append({"rang": rang, "titel": info["titel"], "error": str(e)})
                # Minimaldaten einfügen
                filme.append({"imdb_id": imdb_id, "rang": rang, "titel": info["titel"]})

            time.sleep(1)  # Rate limiting

        browser.close()

    # Sortiere nach Rang
    filme.sort(key=lambda x: x.get("rang", 999))

    # Speichern
    result = {
        "metadata": {
            "description": "Deterministische Ground-Truth für IMDb Top 250 via CSS-Selektoren",
            "quellen": "IMDb.com, deterministisch extrahiert April 2026",
            "felder": ["titel", "jahr", "bewertung", "dauer_minuten", "regisseur", "genres"],
            "n_filme": len(filme),
            "n_errors": len(errors),
        },
        "filme": filme,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"\nGespeichert: {output_path}")
    logger.info(f"Filme: {len(filme)}, Fehler: {len(errors)}")

    if errors:
        logger.warning("Fehler bei:")
        for e in errors:
            logger.warning(f"  Rang {e['rang']}: {e.get('titel', '?')} — {e.get('missing', e.get('error', '?'))}")


if __name__ == "__main__":
    main()
