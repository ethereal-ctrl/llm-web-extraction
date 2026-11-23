"""
IMDb Spider - Web Scraper für IMDb Top 250
"""
import logging
import time
import re
from typing import List, Dict, Any, Optional, Generator
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright, Browser, Page
from bs4 import BeautifulSoup

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from config import SCRAPER_CONFIG
from scraper.cleaner import HTMLCleaner

logger = logging.getLogger(__name__)


class IMDbScraper:
    """
    Web Scraper für IMDb Top 250 Filme.

    Crawlt die Top 250 Liste und einzelne Filmseiten.
    """

    def __init__(
        self,
        max_movies: int = None,
        delay: float = None,
    ):
        """
        Initialisiert den IMDb Scraper.

        Args:
            max_movies: Maximale Anzahl zu scrapender Filme
            delay: Verzögerung zwischen Requests in Sekunden
        """
        self.base_url = SCRAPER_CONFIG["base_url"]
        self.top_250_url = SCRAPER_CONFIG["top_250_url"]
        self.max_movies = max_movies or SCRAPER_CONFIG["max_movies"]
        self.delay = delay or SCRAPER_CONFIG["delay_seconds"]

        # Playwright Browser
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.context = self.browser.new_context(
            user_agent=SCRAPER_CONFIG["user_agent"],
            locale="en-US",
        )
        self.page = self.context.new_page()

        # HTML Cleaner
        self.cleaner = HTMLCleaner()

        logger.info(
            f"IMDb Scraper initialisiert: max_movies={self.max_movies}, delay={self.delay}s"
        )

    def fetch_page(self, url: str) -> Optional[str]:
        """
        Lädt eine Webseite mit Playwright (JavaScript-Rendering).

        Args:
            url: URL der Seite

        Returns:
            HTML-Inhalt oder None bei Fehler
        """
        try:
            logger.debug(f"Lade: {url}")
            self.page.goto(url, wait_until="domcontentloaded", timeout=30000)

            # Warte auf Hauptinhalt
            self.page.wait_for_selector('h1, [data-testid="hero__pageTitle"]', timeout=10000)

            # Rate Limiting
            time.sleep(self.delay)

            return self.page.content()

        except Exception as e:
            logger.error(f"Fehler beim Laden von {url}: {e}")
            return None

    def get_top_250_urls(self) -> List[Dict[str, Any]]:
        """
        Extrahiert die Film-URLs aus der Top 250 Liste.

        Returns:
            Liste von Dictionaries mit URL, Rang und Basis-Infos
        """
        logger.info(f"Lade IMDb Top 250 Liste...")

        html = self.fetch_page(self.top_250_url)
        if not html:
            logger.error("Konnte Top 250 Liste nicht laden")
            return []

        soup = BeautifulSoup(html, "lxml")
        movies = []

        # IMDb Top 250 Struktur (Stand 2024)
        # Suche nach Film-Containern
        movie_items = soup.select('li.ipc-metadata-list-summary-item')

        if not movie_items:
            # Alternative Selektoren versuchen
            movie_items = soup.select('.lister-list tr, .chart tbody tr')

        if not movie_items:
            logger.warning("Keine Film-Items gefunden, versuche alternativen Parser")
            return self._parse_top_250_alternative(soup)

        for idx, item in enumerate(movie_items, 1):
            if idx > self.max_movies:
                break

            try:
                movie_data = self._parse_movie_item(item, idx)
                if movie_data:
                    movies.append(movie_data)
                    logger.debug(f"[{idx}] {movie_data.get('titel', 'Unknown')}")

            except Exception as e:
                logger.warning(f"Fehler bei Film {idx}: {e}")
                continue

        logger.info(f"{len(movies)} Filme aus Top 250 extrahiert")
        return movies

    def _parse_movie_item(self, item, rang: int) -> Optional[Dict[str, Any]]:
        """
        Parst ein Film-Item aus der Liste.

        Args:
            item: BeautifulSoup Element
            rang: Rang in der Liste

        Returns:
            Film-Daten oder None
        """
        # Link zum Film finden
        link = item.select_one('a[href*="/title/"]')
        if not link:
            return None

        href = link.get("href", "")
        url = urljoin(self.base_url, href.split("?")[0])

        # IMDb ID extrahieren
        imdb_id_match = re.search(r'/title/(tt\d+)', href)
        imdb_id = imdb_id_match.group(1) if imdb_id_match else None

        if not imdb_id:
            return None

        # Titel (aus Link-Text oder separatem Element)
        titel = link.get_text(strip=True)
        if not titel:
            title_elem = item.select_one('h3, .titleColumn a, .ipc-title__text')
            titel = title_elem.get_text(strip=True) if title_elem else f"Film {rang}"

        # Rang aus Titel entfernen (z.B. "1. The Shawshank Redemption")
        titel = re.sub(r'^\d+\.\s*', '', titel)

        # Bewertung (falls in Liste vorhanden)
        rating_elem = item.select_one('.ratingColumn strong, .ipc-rating-star--rating')
        bewertung = None
        if rating_elem:
            try:
                bewertung = float(rating_elem.get_text(strip=True))
            except ValueError:
                pass

        # Jahr (falls in Liste vorhanden)
        year_elem = item.select_one('.secondaryInfo, .cli-title-metadata-item')
        jahr = None
        if year_elem:
            year_match = re.search(r'\((\d{4})\)|(\d{4})', year_elem.get_text())
            if year_match:
                jahr = int(year_match.group(1) or year_match.group(2))

        return {
            "rang": rang,
            "imdb_id": imdb_id,
            "url": url,
            "titel": titel,
            "bewertung": bewertung,
            "jahr": jahr,
        }

    def _parse_top_250_alternative(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Alternativer Parser für Top 250 falls Hauptparser fehlschlägt.

        Args:
            soup: BeautifulSoup Objekt

        Returns:
            Liste von Film-Daten
        """
        movies = []

        # Alle Links zu Filmseiten finden
        all_links = soup.select('a[href*="/title/tt"]')

        seen_ids = set()
        rang = 0

        for link in all_links:
            href = link.get("href", "")
            imdb_id_match = re.search(r'/title/(tt\d+)', href)

            if not imdb_id_match:
                continue

            imdb_id = imdb_id_match.group(1)

            # Duplikate vermeiden
            if imdb_id in seen_ids:
                continue

            seen_ids.add(imdb_id)
            rang += 1

            if rang > self.max_movies:
                break

            url = urljoin(self.base_url, f"/title/{imdb_id}/")
            titel = link.get_text(strip=True) or f"Film {rang}"
            titel = re.sub(r'^\d+\.\s*', '', titel)

            movies.append({
                "rang": rang,
                "imdb_id": imdb_id,
                "url": url,
                "titel": titel,
                "bewertung": None,
                "jahr": None,
            })

        logger.info(f"Alternativer Parser: {len(movies)} Filme gefunden")
        return movies

    def fetch_movie_detail(self, url: str) -> Optional[str]:
        """
        Lädt die Detailseite eines Films.

        Args:
            url: URL der Filmseite

        Returns:
            Bereinigter HTML-Inhalt
        """
        html = self.fetch_page(url)
        if not html:
            return None

        # HTML für LLM bereinigen
        cleaned = self.cleaner.clean_for_imdb(html)

        if len(cleaned) < 100:
            logger.warning(f"Sehr wenig Inhalt extrahiert von {url}")
            # Fallback auf normale Bereinigung
            cleaned = self.cleaner.clean(html)

        return cleaned

    def scrape_movies(self) -> Generator[Dict[str, Any], None, None]:
        """
        Generator der alle Filme scrapt.

        Yields:
            Dictionary mit Film-Daten und bereinigtem HTML
        """
        # Top 250 Liste laden
        movie_list = self.get_top_250_urls()

        if not movie_list:
            logger.error("Keine Filme in der Liste gefunden")
            return

        logger.info(f"Starte Scraping von {len(movie_list)} Filmen...")

        for idx, movie in enumerate(movie_list, 1):
            logger.info(f"[{idx}/{len(movie_list)}] Scrape: {movie['titel']}")

            # Detailseite laden
            html_content = self.fetch_movie_detail(movie["url"])

            if html_content:
                yield {
                    **movie,
                    "html_content": html_content,
                }
            else:
                logger.warning(f"Konnte {movie['titel']} nicht laden")
                yield {
                    **movie,
                    "html_content": None,
                    "error": "Laden fehlgeschlagen",
                }

    def close(self):
        """Schließt Browser und Playwright."""
        try:
            self.page.close()
            self.context.close()
            self.browser.close()
            self.playwright.stop()
        except Exception:
            pass
        logger.info("Scraper Session geschlossen")
