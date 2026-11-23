"""
Site-Konfiguration für den generischen Scraper.
Jede Website wird als SiteConfig-Objekt definiert.
"""
from dataclasses import dataclass, field
from typing import List, Any, Optional


@dataclass
class FieldConfig:
    """Konfiguration eines zu extrahierenden Feldes."""
    name: str           # Feldname im Output-JSON
    description: str    # Beschreibung für den LLM-Prompt
    type: str           # "string", "float", "int", "array", "bool"
    example: Any = None # Beispielwert für Few-Shot


@dataclass
class SiteConfig:
    """
    Vollständige Konfiguration einer Website für den generischen Scraper.
    Zum Umschalten zwischen Sites: einfach eine andere SiteConfig übergeben.
    """
    name: str                          # Anzeigename (z.B. "IMDb Top 250")
    list_url: str                      # Start-URL der Listenseite
    item_selector: str                 # CSS-Selektor für einzelne Items auf Listenseite
    fields: List[FieldConfig]          # Felder die extrahiert werden sollen
    pagination_template: Optional[str] = None  # URL-Template mit {page}, z.B. "/page/{page}/"
    max_pages: int = 1                 # Anzahl Seiten die gecrawlt werden
    delay: float = 1.0                 # Sekunden zwischen Requests
    encoding: str = "utf-8"           # Seitenencoding
    ground_truth_path: Optional[str] = None    # Pfad zur Ground-Truth-Datei

    @property
    def field_names(self) -> List[str]:
        return [f.name for f in self.fields]

    def get_prompt_field_list(self) -> str:
        """Generiert die Feldliste für den LLM-Prompt."""
        lines = []
        for f in self.fields:
            example_str = f" (Beispiel: {f.example})" if f.example is not None else ""
            lines.append(f"- {f.name}: {f.description}{example_str}")
        return "\n".join(lines)
