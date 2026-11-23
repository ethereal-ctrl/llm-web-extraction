"""
Generischer Scraper für statische Websites.
Nutzt requests + BeautifulSoup (kein Playwright nötig für statische Seiten).
Zum Umschalten: einfach eine andere SiteConfig übergeben.
"""
import logging
import time
from typing import List, Dict, Any, Optional

import requests
from bs4 import BeautifulSoup

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from sites.site_config import SiteConfig

logger = logging.getLogger(__name__)


class GenericScraper:
    """
    Site-agnostischer Scraper für statische HTML-Seiten.
    Arbeitet mit SiteConfig – kein seitenspezifischer Code nötig.
    """

    def __init__(self, config: SiteConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })
        logger.info(f"GenericScraper initialisiert für: {config.name}")

    def fetch_page(self, url: str) -> Optional[str]:
        """Lädt eine Seite und gibt den HTML-Inhalt zurück."""
        try:
            resp = self.session.get(url, timeout=15)
            resp.encoding = self.config.encoding
            resp.raise_for_status()
            time.sleep(self.config.delay)
            return resp.text
        except Exception as e:
            logger.error(f"Fehler beim Laden von {url}: {e}")
            return None

    def extract_items_from_page(self, html: str) -> List[str]:
        """
        Extrahiert die HTML-Snippets einzelner Items von einer Listenseite.
        Nutzt den item_selector aus der SiteConfig.
        """
        soup = BeautifulSoup(html, "lxml")
        items = soup.select(self.config.item_selector)
        return [str(item) for item in items]

    def scrape_all(self, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Scrapt alle Items über alle konfigurierten Seiten.

        Returns:
            Liste von Dicts mit 'html' (Item-HTML) und 'page' (Seitennummer)
        """
        all_items = []
        total_pages = self.config.max_pages

        for page_num in range(1, total_pages + 1):
            if self.config.pagination_template:
                url = self.config.pagination_template.format(page=page_num)
            else:
                url = self.config.list_url

            logger.info(f"Lade Seite {page_num}/{total_pages}: {url}")
            html = self.fetch_page(url)

            if not html:
                logger.warning(f"Seite {page_num} konnte nicht geladen werden")
                continue

            items = self.extract_items_from_page(html)
            logger.info(f"  {len(items)} Items gefunden")

            for idx, item_html in enumerate(items):
                all_items.append({
                    "id": len(all_items) + 1,
                    "page": page_num,
                    "item_index": idx,
                    "html": item_html,
                })

                if max_items and len(all_items) >= max_items:
                    logger.info(f"max_items={max_items} erreicht")
                    return all_items

        logger.info(f"Gesamt: {len(all_items)} Items von {self.config.name}")
        return all_items

    def generate_ground_truth(self) -> List[Dict[str, Any]]:
        """
        Generiert Ground Truth programmatisch via BeautifulSoup.
        Nur für toscrape.com sinnvoll (deterministisch, bekannte Struktur).
        Wird als Referenz für die Evaluation genutzt.
        """
        raise NotImplementedError("Muss pro Site implementiert werden")
