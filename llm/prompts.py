"""
Prompt Templates für die LLM-basierte Datenextraktion
"""

class PromptTemplates:
    """
    Sammlung von Prompt-Templates für die Film-Datenextraktion.
    """

    # Offizielle IMDb-Spielfilm-Genres (Stand 2026, Filterungsmenge der Pipeline)
    # 22 Einträge: TV-spezifische Genres (Game-Show, News, Reality-TV, Talk-Show)
    # und Short sind ausgeschlossen, weil sie auf Top-250-Spielfilmen nicht
    # auftreten und die Negativ-Filter-Wirkung der Whitelist abschwächen würden.
    # Identisch mit der Liste in scraper/cleaner.py und im JSON-Schema von
    # llm/ollama_client.py.
    VALID_IMDB_GENRES = {
        "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
        "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "History",
        "Horror", "Music", "Musical", "Mystery", "Romance", "Sci-Fi",
        "Sport", "Thriller", "War", "Western",
    }

    SYSTEM_PROMPT = """Du bist ein präziser Datenextraktor für Filminformationen.
Deine Aufgabe ist es, strukturierte Daten aus HTML-Inhalten zu extrahieren.

WICHTIGE REGELN:
1. Gib NUR valides JSON zurück - keine Erklärungen, kein Markdown, KEINE Kommentare
2. Extrahiere nur Informationen, die im HTML vorhanden sind
3. Verwende null für fehlende Werte
4. Zahlen als numerische Werte (nicht als Strings)
5. Arrays für mehrere Werte (z.B. Genres)
6. Bewertungen als Dezimalzahl (z.B. 9.3, nicht "9.3/10")
7. dauer_minuten: Wenn das Format "2h 22m" vorliegt, rechne in Minuten um (z.B. 142)
8. genres: NUR offizielle IMDb-Genres: Action, Adventure, Animation, Biography, Comedy, Crime, Documentary, Drama, Family, Fantasy, Film-Noir, History, Horror, Music, Musical, Mystery, Romance, Sci-Fi, Sport, Thriller, War, Western. KEINE User-Tags wie "Authentic emotion" oder "Period Drama".
9. besetzung: Array von Schauspielern aus den BESETZUNG-Zeilen, je Eintrag als Objekt {"name": ..., "als_charakter": ...}. Übernimm Namen und Rollen exakt, erfinde keine Schauspieler."""

    EXTRACTION_PROMPT = """Extrahiere die Filminformationen aus diesem HTML.

Gewünschte Felder:
- titel: Filmtitel (String)
- original_titel: Originaltitel falls anders (String oder null)
- jahr: Erscheinungsjahr (Integer)
- bewertung: IMDb Bewertung (Float, z.B. 9.3)
- anzahl_bewertungen: Anzahl der Bewertungen (Integer, z.B. 2800000)
- dauer_minuten: Filmlänge in Minuten als Zahl (Integer, z.B. 142). Falls das Format "2h 22m" vorliegt, rechne in Minuten um.
- regisseur: Name des Regisseurs (String, nur der Regisseur – nicht Drehbuchautor oder Produzent)
- genres: Nur die offiziellen IMDb-Genres (Array von Strings, z.B. ["Drama", "Crime"]). Keine User-Tags, Keywords oder Beschreibungen wie "Authentic emotion", "Period Drama" etc.
- beschreibung: Kurze Filmbeschreibung/Plot (String)

HTML-INHALT:
{html_content}

Antworte NUR mit dem JSON-Objekt:"""

    EXTRACTION_PROMPT_MINIMAL = """Extrahiere Filmdaten aus diesem Text und gib JSON zurück.

Felder: titel, jahr, bewertung, regisseur, genres (Array), beschreibung, dauer_minuten

TEXT:
{text_content}

JSON:"""

    @classmethod
    def get_extraction_prompt(cls, html_content: str, minimal: bool = False) -> str:
        """
        Generiert den Extraktions-Prompt.

        Args:
            html_content: Bereinigter HTML/Text-Inhalt
            minimal: Minimaler Prompt für kleinere Modelle

        Returns:
            Formatierter Prompt
        """
        if minimal:
            return cls.EXTRACTION_PROMPT_MINIMAL.format(text_content=html_content)
        return cls.EXTRACTION_PROMPT.format(html_content=html_content)

    @classmethod
    def get_system_prompt(cls) -> str:
        """Gibt den System-Prompt zurück."""
        return cls.SYSTEM_PROMPT

    # Beispiel-Output für Few-Shot Learning
    EXAMPLE_OUTPUT = """{
  "titel": "The Shawshank Redemption",
  "original_titel": null,
  "jahr": 1994,
  "bewertung": 9.3,
  "anzahl_bewertungen": 2800000,
  "dauer_minuten": 142,
  "regisseur": "Frank Darabont",
  "genres": ["Drama"],
  "besetzung": [
    {"name": "Tim Robbins", "als_charakter": "Andy Dufresne"},
    {"name": "Morgan Freeman", "als_charakter": "Ellis Boyd 'Red' Redding"},
    {"name": "Bob Gunton", "als_charakter": "Warden Norton"}
  ],
  "beschreibung": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency."
}"""

    FEW_SHOT_PROMPT = """Extrahiere die Filminformationen aus diesem HTML.

WICHTIG:
- genres: NUR offizielle IMDb-Genres aus dieser Liste: Action, Adventure, Animation, Biography, Comedy, Crime, Documentary, Drama, Family, Fantasy, Film-Noir, History, Horror, Music, Musical, Mystery, Romance, Sci-Fi, Sport, Thriller, War, Western. Ignoriere ALLE anderen Tags/Keywords wie "Authentic emotion", "Period Drama", "Prison Drama", "Thought-provoking" etc.
- dauer_minuten: Als Zahl in Minuten (z.B. "2h 22m" → 142). Suche nach LAUFZEIT oder im Metadaten-String.
- regisseur: Nur der Regisseur, nicht Drehbuchautoren oder Produzenten
- besetzung: Array mit ALLEN Schauspielern aus den BESETZUNG-Zeilen, je Eintrag {{"name": ..., "als_charakter": ...}}. Gib JEDE einzelne BESETZUNG-Zeile als eigenen Eintrag aus, kürze die Liste NICHT. Übernimm Name und Rolle exakt wie angegeben, erfinde keine Schauspieler.
- Antworte NUR mit validem JSON, KEINE Kommentare (// oder /* */)

BEISPIEL-AUSGABE:
{example}

HTML-INHALT:
{html_content}

JSON:"""

    @classmethod
    def get_few_shot_prompt(cls, html_content: str) -> str:
        """
        Generiert einen Few-Shot Prompt mit Beispiel.

        Args:
            html_content: HTML-Inhalt

        Returns:
            Formatierter Prompt mit Beispiel
        """
        return cls.FEW_SHOT_PROMPT.format(
            example=cls.EXAMPLE_OUTPUT,
            html_content=html_content
        )
