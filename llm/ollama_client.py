"""
Ollama Client für lokale LLM-Inferenz
"""
import json
import logging
import re
from typing import Dict, Any, Optional

import ollama
from ollama import Client, ResponseError

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from config import OLLAMA_CONFIG
from llm.prompts import PromptTemplates

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client für die Kommunikation mit dem lokalen Ollama LLM.
    """

    def __init__(
        self,
        host: str = None,
        model: str = None,
        timeout: int = None,
    ):
        """
        Initialisiert den Ollama Client.

        Args:
            host: Ollama Server URL
            model: Modellname (z.B. "llama3.1:8b")
            timeout: Timeout in Sekunden
        """
        self.host = host or OLLAMA_CONFIG["host"]
        self.model = model or OLLAMA_CONFIG["model"]
        self.timeout = timeout or OLLAMA_CONFIG["timeout"]
        self.options = OLLAMA_CONFIG.get("options", {})

        # Ollama Client initialisieren
        self.client = Client(host=self.host, timeout=self.timeout)

        logger.info(f"Ollama Client initialisiert: {self.host}, Modell: {self.model}")

    def is_available(self) -> bool:
        """
        Prüft ob Ollama Server und Modell verfügbar sind.

        Returns:
            True wenn verfügbar
        """
        try:
            # Server erreichbar?
            models = self.client.list()

            # Modell vorhanden?
            model_names = [m.model for m in models.models]
            base_model = self.model.split(":")[0]

            if not any(base_model in name for name in model_names):
                logger.warning(
                    f"Modell '{self.model}' nicht gefunden. "
                    f"Verfügbar: {model_names}"
                )
                return False

            logger.info(f"Ollama verfügbar. Modell: {self.model}")
            return True

        except Exception as e:
            logger.error(f"Ollama nicht erreichbar: {e}")
            return False

    def pull_model(self) -> bool:
        """
        Lädt das Modell herunter falls nicht vorhanden.

        Returns:
            True wenn erfolgreich
        """
        try:
            logger.info(f"Lade Modell: {self.model}...")
            self.client.pull(self.model)
            logger.info(f"Modell {self.model} erfolgreich geladen.")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells: {e}")
            return False

    def extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extrahiert JSON aus der LLM-Antwort.

        Args:
            text: Rohe LLM-Antwort

        Returns:
            Parsed JSON oder None
        """
        # JavaScript-Kommentare entfernen (// ... und /* ... */)
        # Manche LLMs schreiben Kommentare in JSON-Ausgaben
        # Ignoriere // in URLs (nach : oder Zeilenstart mit ")
        text_clean = re.sub(r'(?<![":])//[^\n"]*(?=\n|$)', '', text)
        text_clean = re.sub(r'/\*.*?\*/', '', text_clean, flags=re.DOTALL)
        # Trailing Kommas vor } oder ] entfernen (häufiger LLM-Fehler)
        text_clean = re.sub(r',\s*([}\]])', r'\1', text_clean)

        # Versuche direkt zu parsen
        try:
            return json.loads(text_clean.strip())
        except json.JSONDecodeError:
            pass

        # Fallback: Original-Text ohne Kommentarbereinigung
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        text = text_clean  # Für alle weiteren Strategien bereinigten Text verwenden

        # Suche nach JSON-Block
        patterns = [
            r'```json\s*(.*?)\s*```',  # Markdown JSON Block
            r'```\s*(.*?)\s*```',       # Markdown Code Block
            r'\{[^{}]*\}',              # Einfaches JSON Objekt
            r'\{.*\}',                  # Beliebiges JSON (greedy)
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str.strip())
                except json.JSONDecodeError:
                    continue

        logger.warning(f"Konnte kein JSON extrahieren aus: {text[:200]}...")
        return None

    def extract_film_data(
        self,
        html_content: str,
        use_few_shot: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Extrahiert Filmdaten aus HTML mittels LLM.

        Args:
            html_content: Bereinigter HTML-Inhalt
            use_few_shot: Few-Shot Prompt verwenden

        Returns:
            Dictionary mit Filmdaten oder None
        """
        try:
            # Prompt generieren
            if use_few_shot:
                user_prompt = PromptTemplates.get_few_shot_prompt(html_content)
            else:
                user_prompt = PromptTemplates.get_extraction_prompt(html_content)

            # LLM aufrufen mit JSON-Modus
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": PromptTemplates.get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                format="json",
                options=self.options,
            )

            # Antwort verarbeiten
            raw_response = response.message.content
            logger.debug(f"LLM Antwort: {raw_response[:500]}...")

            # JSON extrahieren
            film_data = self.extract_json(raw_response)

            if film_data:
                # Datentypen validieren/konvertieren
                film_data = self._validate_film_data(film_data)
                logger.info(f"Film extrahiert: {film_data.get('titel', 'Unknown')}")
                return film_data
            else:
                logger.warning("Keine validen Filmdaten extrahiert.")
                return None

        except ResponseError as e:
            logger.error(f"Ollama API Fehler: {e}")
            return None
        except Exception as e:
            logger.error(f"Fehler bei der Extraktion: {e}")
            return None

    # JSON-Schema für Film-Extraktion (für strukturierte Ausgabe via Ollama).
    # Entspricht den Validierungsregeln in _validate_film_data, aber wird
    # auf Decoder-Ebene durch Token-Masking erzwungen (XGrammar/Outlines-Prinzip).
    FILM_JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "titel":              {"type": ["string", "null"]},
            "original_titel":     {"type": ["string", "null"]},
            "jahr":               {"type": ["integer", "null"]},
            "bewertung":          {"type": ["number", "null"]},
            "anzahl_bewertungen": {"type": ["integer", "null"]},
            "dauer_minuten":      {"type": ["integer", "null"]},
            "regisseur":          {"type": ["string", "null"]},
            "genres": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "Action", "Adventure", "Animation", "Biography",
                        "Comedy", "Crime", "Documentary", "Drama", "Family",
                        "Fantasy", "Film-Noir", "History", "Horror", "Music",
                        "Musical", "Mystery", "Romance", "Sci-Fi", "Sport",
                        "Thriller", "War", "Western",
                    ],
                },
            },
            "besetzung": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name":          {"type": "string"},
                        "als_charakter": {"type": ["string", "null"]},
                    },
                    "required": ["name"],
                },
            },
            "beschreibung": {"type": ["string", "null"]},
        },
        "required": [
            "titel", "jahr", "bewertung", "dauer_minuten",
            "regisseur", "genres",
        ],
    }

    def extract_film_data_schema(
        self,
        html_content: str,
        use_few_shot: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Extraktion mit JSON-Schema-Constraints (Decoder-Token-Masking).

        Verwendet `format=<schema_dict>` statt `format="json"`. Der Decoder
        akzeptiert je Schritt nur Tokens, die das Schema fortsetzen können.
        Damit entfällt die sechsstufige Fallback-Kaskade — jede Ausgabe ist
        per Konstruktion strukturell und typsicher konform.

        Returns:
            Validierte Filmdaten oder None bei Fehlern auf Modellseite.
        """
        try:
            if use_few_shot:
                user_prompt = PromptTemplates.get_few_shot_prompt(html_content)
            else:
                user_prompt = PromptTemplates.get_extraction_prompt(html_content)

            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": PromptTemplates.get_system_prompt()},
                    {"role": "user", "content": user_prompt},
                ],
                format=self.FILM_JSON_SCHEMA,
                options=self.options,
            )

            raw_response = response.message.content
            logger.debug(f"LLM (Schema) Antwort: {raw_response[:500]}...")

            # Schema garantiert: direkter Parse muss klappen, sonst echtes
            # Modell-/Server-Problem. Trotzdem defensiv extrahieren.
            try:
                film_data = json.loads(raw_response.strip())
            except json.JSONDecodeError:
                film_data = self.extract_json(raw_response)

            if film_data:
                film_data = self._validate_film_data(film_data)
                return film_data
            return None

        except ResponseError as e:
            logger.error(f"Ollama Schema-API Fehler: {e}")
            return None
        except Exception as e:
            logger.error(f"Fehler bei der Schema-Extraktion: {e}")
            return None

    def _validate_film_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validiert und bereinigt die extrahierten Filmdaten.

        Args:
            data: Rohe Filmdaten

        Returns:
            Bereinigte Filmdaten
        """
        validated = {}

        # String-Felder
        for field in ["titel", "original_titel", "regisseur", "beschreibung"]:
            value = data.get(field)
            if value and isinstance(value, str):
                validated[field] = value.strip()
            else:
                validated[field] = None

        # Integer-Felder
        for field in ["jahr", "anzahl_bewertungen"]:
            value = data.get(field)
            if value is not None:
                try:
                    if isinstance(value, str):
                        numbers = re.findall(r'\d+', value)
                        validated[field] = int(numbers[0]) if numbers else None
                    else:
                        validated[field] = int(value)
                except (ValueError, TypeError):
                    validated[field] = None
            else:
                validated[field] = None

        # dauer_minuten: "2h 22m" → 142, "142" → 142
        value = data.get("dauer_minuten")
        if value is not None:
            try:
                if isinstance(value, str):
                    hours_match = re.search(r'(\d+)\s*h', value)
                    mins_match = re.search(r'(\d+)\s*m(?:in)?', value)
                    if hours_match or mins_match:
                        hours = int(hours_match.group(1)) if hours_match else 0
                        mins = int(mins_match.group(1)) if mins_match else 0
                        validated["dauer_minuten"] = hours * 60 + mins
                    else:
                        numbers = re.findall(r'\d+', value)
                        validated["dauer_minuten"] = int(numbers[0]) if numbers else None
                else:
                    validated["dauer_minuten"] = int(value)
            except (ValueError, TypeError):
                validated["dauer_minuten"] = None
        else:
            validated["dauer_minuten"] = None

        # Float-Felder
        for field in ["bewertung"]:
            value = data.get(field)
            if value is not None:
                try:
                    if isinstance(value, str):
                        # Entferne "/10" oder ähnliches
                        value = value.replace("/10", "").strip()
                    validated[field] = float(value)
                except (ValueError, TypeError):
                    validated[field] = None
            else:
                validated[field] = None

        # Array-Felder mit Genre-Whitelist-Filter
        genres = data.get("genres", data.get("genre", []))
        if isinstance(genres, str):
            genre_list = [g.strip() for g in genres.split(",")]
        elif isinstance(genres, list):
            genre_list = [str(g).strip() for g in genres if g]
        else:
            genre_list = []

        # Nur offizielle IMDb-Genres behalten
        filtered = [g for g in genre_list if g in PromptTemplates.VALID_IMDB_GENRES]
        if not filtered and genre_list:
            # Fallback: Case-insensitive Matching
            valid_lower = {v.lower(): v for v in PromptTemplates.VALID_IMDB_GENRES}
            filtered = [valid_lower[g.lower()] for g in genre_list if g.lower() in valid_lower]
        if not filtered and genre_list:
            logger.warning(f"Keine offiziellen Genres erkannt in: {genre_list}")
        validated["genres"] = filtered

        # Besetzung (m:n): Liste von {name, als_charakter} normalisieren
        besetzung = data.get("besetzung", data.get("cast", []))
        cast_list = []
        if isinstance(besetzung, list):
            for entry in besetzung:
                if isinstance(entry, dict):
                    name = str(entry.get("name", "")).strip()
                    rolle = entry.get("als_charakter") or entry.get("rolle")
                    rolle = str(rolle).strip() if rolle else None
                elif isinstance(entry, str):
                    # Modell gab nur den Namen zurück
                    name, rolle = entry.strip(), None
                else:
                    continue
                if name:
                    cast_list.append({"name": name, "als_charakter": rolle or None})
        validated["besetzung"] = cast_list

        return validated

    def test_extraction(self) -> bool:
        """
        Testet die Extraktion mit einem Beispiel.

        Returns:
            True wenn erfolgreich
        """
        test_html = """
        <div class="movie">
            <h1>The Matrix</h1>
            <span class="year">1999</span>
            <span class="rating">8.7</span>
            <span class="genre">Action, Sci-Fi</span>
            <span class="director">Lana Wachowski, Lilly Wachowski</span>
            <p class="plot">A computer hacker learns about the true nature of reality.</p>
        </div>
        """

        result = self.extract_film_data(test_html, use_few_shot=True)

        if result and result.get("titel"):
            logger.info(f"Test erfolgreich: {result}")
            return True
        else:
            logger.error("Test fehlgeschlagen")
            return False
