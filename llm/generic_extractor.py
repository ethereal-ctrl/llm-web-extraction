"""
Generischer LLM-Extraktor für beliebige SiteConfigs.
Generiert Prompts dynamisch aus der SiteConfig – kein seitenspezifischer Code.
"""
import json
import logging
import re
import time
from typing import Dict, Any, Optional, List

from ollama import Client, ResponseError

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from config import OLLAMA_CONFIG
from sites.site_config import SiteConfig

logger = logging.getLogger(__name__)


class GenericExtractor:
    """
    Site-agnostischer LLM-Extraktor.
    Prompt wird aus SiteConfig generiert – zum Umschalten einfach andere Config übergeben.
    """

    def __init__(self, config: SiteConfig, model: str = None):
        self.config = config
        self.model = model or OLLAMA_CONFIG["model"]
        self.client = Client(
            host=OLLAMA_CONFIG["host"],
            timeout=OLLAMA_CONFIG["timeout"]
        )
        self.options = OLLAMA_CONFIG.get("options", {})
        logger.info(f"GenericExtractor: {config.name} | Modell: {self.model}")

    def build_system_prompt(self) -> str:
        return (
            "Du bist ein präziser Datenextraktor. "
            "Extrahiere strukturierte Daten aus HTML-Snippets. "
            "Antworte NUR mit validem JSON – keine Erklärungen, kein Markdown, keine Kommentare. "
            "Verwende null für fehlende Werte. "
            "Verwende EXAKT die vorgegebenen Feldnamen – keine Übersetzung, keine Umbenennung."
        )

    def build_extraction_prompt(self, html: str) -> str:
        field_list = self.config.get_prompt_field_list()
        field_names = list(self.config.field_names)
        skeleton = "{" + ", ".join(f'"{n}": ...' for n in field_names) + "}"
        return (
            f"Extrahiere die folgenden Felder aus diesem HTML-Snippet:\n\n"
            f"{field_list}\n\n"
            f"Antworte mit einem JSON-Objekt mit EXAKT diesen Schlüsseln:\n"
            f"{skeleton}\n\n"
            f"HTML:\n{html}\n\n"
            f"JSON:"
        )

    def extract_json(self, text: str) -> Optional[Dict]:
        """Extrahiert JSON aus LLM-Antwort (robust gegen Formatfehler)."""
        # Kommentare entfernen
        text = re.sub(r'(?<![":])//[^\n"]*', '', text)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        text = re.sub(r',\s*([}\]])', r'\1', text)

        for attempt in [text.strip(), text]:
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                pass

        for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```', r'\{.*\}']:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1) if '```' in pattern else m.group(0))
                except json.JSONDecodeError:
                    continue
        return None

    def postprocess(self, data: Dict) -> Dict:
        """Typkonvertierung basierend auf SiteConfig-Feldtypen."""
        # Normalisiere Keys: lowercase ohne Leerzeichen für Fallback-Lookup
        data_lower = {k.lower().replace(" ", "_"): v for k, v in data.items()}

        result = {}
        for field in self.config.fields:
            # Erst exakter Match, dann case-insensitive Fallback
            val = data.get(field.name)
            if val is None:
                val = data_lower.get(field.name.lower())
            try:
                if val is None:
                    result[field.name] = None
                elif field.type == "int":
                    if isinstance(val, str):
                        # "Three" → 3 für Sternebewertungen
                        word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
                        if val.lower() in word_map:
                            result[field.name] = word_map[val.lower()]
                        else:
                            nums = re.findall(r'\d+', val)
                            result[field.name] = int(nums[0]) if nums else None
                    else:
                        result[field.name] = int(val)
                elif field.type == "float":
                    if isinstance(val, str):
                        # "£51.77" → 51.77
                        nums = re.findall(r'[\d.]+', val)
                        result[field.name] = float(nums[0]) if nums else None
                    else:
                        result[field.name] = float(val)
                elif field.type == "bool":
                    if isinstance(val, bool):
                        result[field.name] = val
                    elif isinstance(val, str):
                        result[field.name] = val.lower() in ("true", "in stock", "yes", "1")
                    else:
                        result[field.name] = bool(val)
                elif field.type == "array":
                    if isinstance(val, list):
                        result[field.name] = [str(v).strip() for v in val]
                    elif isinstance(val, str):
                        result[field.name] = [v.strip() for v in val.split(",")]
                    else:
                        result[field.name] = []
                else:  # string
                    result[field.name] = str(val).strip() if val else None
            except Exception:
                result[field.name] = None
        return result

    def extract(self, html: str) -> Optional[Dict[str, Any]]:
        """Extrahiert ein einzelnes Item aus HTML."""
        try:
            t0 = time.time()
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.build_system_prompt()},
                    {"role": "user", "content": self.build_extraction_prompt(html)},
                ],
                format="json",
                options=self.options,
            )
            elapsed = time.time() - t0

            raw = response.message.content
            token_info = {
                "prompt_tokens": getattr(response, "prompt_eval_count", None),
                "response_tokens": getattr(response, "eval_count", None),
            }

            data = self.extract_json(raw)
            if not data:
                logger.warning("JSON-Extraktion fehlgeschlagen")
                return None

            result = self.postprocess(data)
            result["_meta"] = {
                "elapsed_s": round(elapsed, 2),
                **token_info,
            }
            return result

        except ResponseError as e:
            logger.error(f"Ollama API Fehler: {e}")
            return None
        except Exception as e:
            logger.error(f"Extraktionsfehler: {e}")
            return None

    def extract_batch(
        self,
        items: List[Dict],
        max_items: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extrahiert eine Liste von Items (aus GenericScraper.scrape_all()).

        Args:
            items: Liste von Dicts mit 'html', 'id', 'page'
            max_items: Maximale Anzahl

        Returns:
            Liste mit extrahierten Daten + Metadaten
        """
        results = []
        total = min(len(items), max_items) if max_items else len(items)

        for i, item in enumerate(items[:total], 1):
            logger.info(f"[{i}/{total}] Extrahiere Item {item['id']} (Seite {item['page']})")
            extracted = self.extract(item["html"])

            if extracted:
                extracted["_id"] = item["id"]
                extracted["_page"] = item["page"]
                extracted["_item_index"] = item["item_index"]
                results.append(extracted)
            else:
                logger.warning(f"Item {item['id']} nicht extrahiert")
                results.append({
                    "_id": item["id"],
                    "_page": item["page"],
                    "_item_index": item["item_index"],
                    "_error": "Extraktion fehlgeschlagen",
                    **{f.name: None for f in self.config.fields},
                    "_meta": {"elapsed_s": 0},
                })

        return results
