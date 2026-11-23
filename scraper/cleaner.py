"""
HTML Cleaner - Bereinigt HTML für LLM-Verarbeitung
"""
import re
import logging
from typing import Optional
from bs4 import BeautifulSoup, Comment

logger = logging.getLogger(__name__)


class HTMLCleaner:
    """
    Bereinigt HTML-Inhalte für die LLM-basierte Extraktion.

    Entfernt unnötige Elemente und behält nur relevante Inhalte,
    um Token zu sparen und die Extraktion zu verbessern.
    """

    # Tags die komplett entfernt werden
    REMOVE_TAGS = [
        "script",
        "style",
        "noscript",
        "iframe",
        "svg",
        "path",
        "meta",
        "link",
        "head",
        "footer",
        "nav",
        "aside",
        "form",
        "input",
        "button",
        "select",
        "textarea",
        "advertisement",
        "ads",
    ]

    # Relevante Selektoren für IMDb Filmseiten
    IMDB_SELECTORS = [
        '[data-testid="hero__pageTitle"]',           # Titel
        '[data-testid="hero-rating-bar__aggregate-rating"]',  # Bewertung
        '[data-testid="title-details-releasedate"]', # Jahr (neu)
        '[data-testid="title-techspec_runtime"]',    # Laufzeit (neu)
        '[data-testid="genres"]',                    # Genres
        '.ipc-chip__text',                           # Genre-Chips
        '[data-testid="plot"]',                      # Plot
        '[data-testid="title-pc-principal-credit"]', # Credits
        '.ipc-metadata-list',                        # Metadata
    ]

    # Offizielle IMDb-Spielfilm-Genres (22 Einträge, identisch mit
    # PromptTemplates.VALID_IMDB_GENRES und JSON-Schema-Enum in
    # OllamaClient.FILM_JSON_SCHEMA).
    VALID_GENRES = {
        "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
        "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "History",
        "Horror", "Music", "Musical", "Mystery", "Romance", "Sci-Fi",
        "Sport", "Thriller", "War", "Western",
    }

    def __init__(self, max_length: int = 8000):
        """
        Initialisiert den HTML Cleaner.

        Args:
            max_length: Maximale Textlänge für LLM
        """
        self.max_length = max_length

    def clean(self, html: str) -> str:
        """
        Bereinigt HTML für LLM-Verarbeitung.

        Args:
            html: Roher HTML-String

        Returns:
            Bereinigter Text/HTML
        """
        if not html:
            return ""

        try:
            soup = BeautifulSoup(html, "lxml")

            # Kommentare entfernen
            for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
                comment.extract()

            # Unnötige Tags entfernen
            for tag in self.REMOVE_TAGS:
                for element in soup.find_all(tag):
                    element.decompose()

            # Leere Tags entfernen
            for tag in soup.find_all():
                if not tag.get_text(strip=True) and tag.name not in ["img", "br"]:
                    tag.decompose()

            # Attribute bereinigen
            for tag in soup.find_all(True):
                # Behalte nur wichtige Attribute
                attrs_to_keep = {}
                for attr in ["href", "src", "alt", "title", "data-testid"]:
                    if attr in tag.attrs:
                        attrs_to_keep[attr] = tag.attrs[attr]
                tag.attrs = attrs_to_keep

            # Text extrahieren mit Struktur
            text = self._extract_structured_text(soup)

            # Whitespace normalisieren
            text = self._normalize_whitespace(text)

            # Auf maximale Länge kürzen
            if len(text) > self.max_length:
                text = text[:self.max_length] + "..."
                logger.debug(f"Text auf {self.max_length} Zeichen gekürzt")

            return text

        except Exception as e:
            logger.error(f"Fehler beim Bereinigen: {e}")
            return html[:self.max_length] if len(html) > self.max_length else html

    def clean_for_imdb(self, html: str) -> str:
        """
        Speziell optimierte Bereinigung für IMDb-Seiten.

        Args:
            html: Roher HTML-String

        Returns:
            Bereinigter, strukturierter Text
        """
        if not html:
            return ""

        try:
            soup = BeautifulSoup(html, "lxml")

            # Relevante Inhalte extrahieren
            relevant_content = []

            # Titel
            title_elem = soup.select_one('[data-testid="hero__pageTitle"], h1')
            if title_elem:
                relevant_content.append(f"TITEL: {title_elem.get_text(strip=True)}")

            # Jahr aus Release-Datum (neues IMDb-Layout)
            release = soup.select_one(
                '[data-testid="title-details-releasedate"] '
                '.ipc-metadata-list-item__list-content-item'
            )
            if release:
                year_match = re.search(r'\b(19|20)\d{2}\b', release.get_text())
                if year_match:
                    relevant_content.append(f"JAHR: {year_match.group(0)}")

            # Fallback Jahr: altes Metadaten-Layout
            if not any("JAHR:" in c for c in relevant_content):
                metadata = soup.select_one('[data-testid="hero-title-block__metadata"]')
                if metadata:
                    meta_text = metadata.get_text(' | ', strip=True)
                    year_match = re.search(r'\b(19|20)\d{2}\b', meta_text)
                    if year_match:
                        relevant_content.append(f"JAHR: {year_match.group(0)}")
                    relevant_content.append(f"METADATEN: {meta_text}")

            # Laufzeit aus techspec (neues Layout)
            runtime = soup.select_one(
                '[data-testid="title-techspec_runtime"] '
                '.ipc-metadata-list-item__content-container'
            )
            if runtime:
                rt_text = runtime.get_text(strip=True)
                relevant_content.append(f"LAUFZEIT: {rt_text}")
                rt_match = re.search(r'(\d+)\s*h\s*(\d+)\s*m', rt_text)
                if rt_match:
                    relevant_content.append(
                        f"LAUFZEIT_MIN: {int(rt_match.group(1)) * 60 + int(rt_match.group(2))}"
                    )
                else:
                    h_match = re.search(r'(\d+)\s*h', rt_text)
                    m_match = re.search(r'(\d+)\s*m', rt_text)
                    total = 0
                    if h_match:
                        total += int(h_match.group(1)) * 60
                    if m_match:
                        total += int(m_match.group(1))
                    if total > 0:
                        relevant_content.append(f"LAUFZEIT_MIN: {total}")

            # Bewertung
            rating = soup.select_one('[data-testid="hero-rating-bar__aggregate-rating__score"]')
            if rating:
                relevant_content.append(f"BEWERTUNG: {rating.get_text(strip=True)}")

            # Anzahl Bewertungen
            rating_count = soup.select_one('[data-testid="hero-rating-bar__aggregate-rating__score"] + div')
            if rating_count:
                relevant_content.append(f"ANZAHL: {rating_count.get_text(strip=True)}")

            # Genres (aus Chip-Elementen, gefiltert auf valide IMDb-Genres)
            chip_elems = soup.select('.ipc-chip__text')
            if chip_elems:
                all_chips = [c.get_text(strip=True) for c in chip_elems]
                valid_genres = [g for g in all_chips if g in self.VALID_GENRES]
                if valid_genres:
                    relevant_content.append(f"GENRES: {', '.join(valid_genres)}")
            # Fallback: altes Genre-Layout
            if not any("GENRES:" in c for c in relevant_content):
                genres = soup.select('[data-testid="genres"] a')
                if genres:
                    genre_list = [g.get_text(strip=True) for g in genres]
                    relevant_content.append(f"GENRES: {', '.join(genre_list)}")

            # Plot/Beschreibung
            plot = soup.select_one('[data-testid="plot"] span, [data-testid="plot-xs_to_m"]')
            if plot:
                relevant_content.append(f"BESCHREIBUNG: {plot.get_text(strip=True)}")

            # Regisseur
            directors = soup.select('[data-testid="title-pc-principal-credit"]:first-of-type a')
            if directors:
                director_names = [d.get_text(strip=True) for d in directors]
                relevant_content.append(f"REGISSEUR: {', '.join(director_names)}")

            # Besetzung (m:n: Schauspieler + gespielte Rolle).
            # Auf die 10 Hauptrollen begrenzt, konsistent mit der
            # Ground-Truth-Tiefe (GT_CAST_DEPTH); tiefere Cast-Ränge
            # schwanken zwischen Snapshots am stärksten.
            cast = self.extract_cast(soup, max_cast=10)
            for entry in cast:
                rolle = entry["als_charakter"]
                if rolle:
                    relevant_content.append(
                        f"BESETZUNG: {entry['name']} als {rolle}"
                    )
                else:
                    relevant_content.append(f"BESETZUNG: {entry['name']}")

            # Details-Tabelle
            details = soup.select('.ipc-metadata-list__item')
            for detail in details[:10]:  # Erste 10 Details
                label = detail.select_one('.ipc-metadata-list-item__label')
                value = detail.select_one('.ipc-metadata-list-item__content-container')
                if label and value:
                    relevant_content.append(
                        f"{label.get_text(strip=True)}: {value.get_text(' ', strip=True)}"
                    )

            # Falls strukturierte Extraktion fehlschlägt, Fallback
            if len(relevant_content) < 3:
                logger.warning("Strukturierte Extraktion unvollständig, verwende Fallback")
                return self.clean(html)

            result = "\n".join(relevant_content)

            # Auf maximale Länge begrenzen
            if len(result) > self.max_length:
                result = result[:self.max_length]

            return result

        except Exception as e:
            logger.error(f"IMDb-spezifische Bereinigung fehlgeschlagen: {e}")
            return self.clean(html)

    def extract_cast(self, source, max_cast: int = 15) -> list:
        """Extrahiert die Besetzung (Schauspieler + gespielte Rolle).

        Bildet die m:n-Beziehung Film <-> Person ab. Jeder Eintrag ist
        ein Dict ``{"name": str, "als_charakter": Optional[str]}``. Die
        Reihenfolge entspricht der Cast-Reihenfolge auf der IMDb-Seite
        (Hauptrollen zuerst).

        Args:
            source: HTML-String oder ein bereits geparstes BeautifulSoup-Objekt.
            max_cast: Maximale Anzahl Cast-Einträge (IMDb zeigt initial 18).

        Returns:
            Liste von Cast-Dicts in DOM-Reihenfolge.
        """
        if isinstance(source, BeautifulSoup):
            soup = source
        else:
            if not source:
                return []
            soup = BeautifulSoup(source, "lxml")

        cast = []
        items = soup.select('[data-testid="title-cast-item"]')
        for item in items[:max_cast]:
            actor_el = item.select_one('[data-testid="title-cast-item__actor"]')
            if not actor_el:
                continue
            name = actor_el.get_text(strip=True)
            if not name:
                continue
            char_el = item.select_one('[data-testid="cast-item-characters-list"]')
            als_charakter = char_el.get_text(" / ", strip=True) if char_el else None
            cast.append({"name": name, "als_charakter": als_charakter or None})
        return cast

    def _extract_structured_text(self, soup: BeautifulSoup) -> str:
        """
        Extrahiert Text mit semantischer Struktur.

        Args:
            soup: BeautifulSoup Objekt

        Returns:
            Strukturierter Text
        """
        lines = []

        # Überschriften
        for tag in ["h1", "h2", "h3"]:
            for elem in soup.find_all(tag):
                text = elem.get_text(strip=True)
                if text:
                    lines.append(f"[{tag.upper()}] {text}")

        # Hauptinhalt
        main_content = soup.find("main") or soup.find("body") or soup
        if main_content:
            text = main_content.get_text(separator="\n", strip=True)
            lines.append(text)

        return "\n".join(lines)

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalisiert Whitespace im Text.

        Args:
            text: Eingabetext

        Returns:
            Bereinigter Text
        """
        # Mehrfache Leerzeichen zu einem
        text = re.sub(r' +', ' ', text)
        # Mehrfache Newlines zu einem
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Whitespace am Zeilenanfang/-ende
        text = "\n".join(line.strip() for line in text.split("\n"))
        return text.strip()

    def extract_text_only(self, html: str) -> str:
        """
        Extrahiert nur den reinen Text ohne HTML.

        Args:
            html: HTML-String

        Returns:
            Nur Text
        """
        soup = BeautifulSoup(html, "lxml")

        # Skripte und Styles entfernen
        for tag in soup.find_all(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return self._normalize_whitespace(text)
