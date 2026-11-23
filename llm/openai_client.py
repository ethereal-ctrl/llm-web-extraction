"""
OpenAI Client für Cloud-LLM-Inferenz

Spiegelt die Schnittstelle von OllamaClient (extract_film_data / extract_json /
_validate_film_data), damit lokale und Cloud-Extraktion unter identischen
Prompts, identischer Fallback-Kaskade und identischer Validierung verglichen
werden können. Einziger Unterschied: die Inferenz läuft über die OpenAI-API
statt über einen lokalen Ollama-Server.
"""
import json
import logging
import os
import re
from typing import Dict, Any, Optional

from openai import OpenAI

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from llm.prompts import PromptTemplates

logger = logging.getLogger(__name__)

# Preise in USD pro 1M Token (Stand der Arbeit; nur für die Kostenschätzung
# im Ergebnis-JSON, ohne Einfluss auf die Extraktion).
PRICING_USD_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o":      {"input": 2.50, "output": 10.00},
}


class OpenAIClient:
    """
    Client für die Kommunikation mit der OpenAI-API.

    Verhält sich nach außen wie OllamaClient: gleiche Prompts, gleiche
    JSON-Fallback-Kaskade, gleiche Validierung inklusive Genre-Whitelist.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = None,
        timeout: int = 120,
        temperature: float = 0.1,
    ):
        """
        Initialisiert den OpenAI Client.

        Args:
            model: Modellname (z.B. "gpt-4o-mini", "gpt-4o")
            api_key: API-Key; default aus Umgebungsvariable OPENAI_API_KEY
            timeout: Timeout in Sekunden
            temperature: Sampling-Temperatur (0.1 wie im lokalen Setup)
        """
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY nicht gesetzt. Trage ihn in die .env ein "
                "(OPENAI_API_KEY=sk-...) oder exportiere ihn in der Shell."
            )

        self.client = OpenAI(api_key=key, timeout=timeout)

        # Token-Verbrauch über alle Aufrufe hinweg (für Kostenschätzung)
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

        logger.info(f"OpenAI Client initialisiert. Modell: {self.model}")

    def is_available(self) -> bool:
        """
        Prüft, ob API-Key und Modell nutzbar sind (ein minimaler Testaufruf).

        Returns:
            True wenn verfügbar
        """
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            logger.info(f"OpenAI erreichbar. Modell: {self.model}")
            return True
        except Exception as e:
            logger.error(f"OpenAI nicht erreichbar: {e}")
            return False

    def extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extrahiert JSON aus der LLM-Antwort.

        Identische sechsstufige Fallback-Kaskade wie OllamaClient.extract_json,
        damit der Vergleich nicht durch unterschiedliches Parsing verzerrt wird.

        Args:
            text: Rohe LLM-Antwort

        Returns:
            Parsed JSON oder None
        """
        if not text:
            return None

        # JavaScript-Kommentare entfernen (// ... und /* ... */),
        # // in URLs bleibt geschützt
        text_clean = re.sub(r'(?<![":])//[^\n"]*(?=\n|$)', '', text)
        text_clean = re.sub(r'/\*.*?\*/', '', text_clean, flags=re.DOTALL)
        # Trailing Kommas vor } oder ] entfernen
        text_clean = re.sub(r',\s*([}\]])', r'\1', text_clean)

        try:
            return json.loads(text_clean.strip())
        except json.JSONDecodeError:
            pass

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        text = text_clean

        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{[^{}]*\}',
            r'\{.*\}',
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

    def _chat(self, user_prompt: str, json_mode: bool = True) -> Optional[str]:
        """Führt den API-Aufruf aus und protokolliert den Token-Verbrauch."""
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": PromptTemplates.get_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        if json_mode:
            # Gegenstück zu format="json" bei Ollama
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)

        if response.usage:
            self.usage["input_tokens"] += response.usage.prompt_tokens
            self.usage["output_tokens"] += response.usage.completion_tokens
        self.usage["calls"] += 1

        return response.choices[0].message.content

    def extract_film_data(
        self,
        html_content: str,
        use_few_shot: bool = True,
        json_mode: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Extrahiert Filmdaten aus HTML mittels OpenAI-Modell.

        Args:
            html_content: Bereinigter HTML-Inhalt
            use_few_shot: Few-Shot Prompt verwenden
            json_mode: response_format=json_object erzwingen

        Returns:
            Dictionary mit Filmdaten oder None
        """
        try:
            if use_few_shot:
                user_prompt = PromptTemplates.get_few_shot_prompt(html_content)
            else:
                user_prompt = PromptTemplates.get_extraction_prompt(html_content)

            raw_response = self._chat(user_prompt, json_mode=json_mode)
            logger.debug(f"LLM Antwort: {(raw_response or '')[:500]}...")

            film_data = self.extract_json(raw_response)

            if film_data:
                film_data = self._validate_film_data(film_data)
                logger.info(f"Film extrahiert: {film_data.get('titel', 'Unknown')}")
                return film_data

            logger.warning("Keine validen Filmdaten extrahiert.")
            return None

        except Exception as e:
            logger.error(f"Fehler bei der Extraktion: {e}")
            return None

    def extract_film_data_raw(
        self,
        html_content: str,
        use_few_shot: bool = True,
        json_mode: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Wie extract_film_data, aber ohne die Validierungsschicht.

        Für Evaluationsskripte, die die Validierung selbst nachbauen
        (z.B. um einzelne Bausteine gezielt abzuschalten).
        """
        if use_few_shot:
            user_prompt = PromptTemplates.get_few_shot_prompt(html_content)
        else:
            user_prompt = PromptTemplates.get_extraction_prompt(html_content)

        raw_response = self._chat(user_prompt, json_mode=json_mode)
        return self.extract_json(raw_response)

    def estimated_cost_usd(self) -> Optional[float]:
        """Schätzt die bisherigen API-Kosten in USD, falls Preise bekannt sind."""
        prices = PRICING_USD_PER_1M.get(self.model)
        if not prices:
            return None
        return (
            self.usage["input_tokens"] / 1_000_000 * prices["input"]
            + self.usage["output_tokens"] / 1_000_000 * prices["output"]
        )

    def _validate_film_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validiert und bereinigt die extrahierten Filmdaten.

        Wortgleich zu OllamaClient._validate_film_data, damit lokale und
        Cloud-Ergebnisse dieselbe Nachverarbeitung durchlaufen.

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

        filtered = [g for g in genre_list if g in PromptTemplates.VALID_IMDB_GENRES]
        if not filtered and genre_list:
            valid_lower = {v.lower(): v for v in PromptTemplates.VALID_IMDB_GENRES}
            filtered = [valid_lower[g.lower()] for g in genre_list
                        if g.lower() in valid_lower]
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
                    name, rolle = entry.strip(), None
                else:
                    continue
                if name:
                    cast_list.append({"name": name, "als_charakter": rolle or None})
        validated["besetzung"] = cast_list

        return validated
