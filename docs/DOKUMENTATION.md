# Technische Dokumentation

## Inhaltsverzeichnis

1. [Projektübersicht](#1-projektübersicht)
2. [Architektur](#2-architektur)
3. [Technologie-Stack](#3-technologie-stack)
4. [Installation & Setup](#4-installation--setup)
5. [Konfiguration](#5-konfiguration)
6. [Modulbeschreibungen](#6-modulbeschreibungen)
7. [LLM-basierte Datenextraktion](#7-llm-basierte-datenextraktion)
8. [Datenbank-Schema](#8-datenbank-schema)
9. [Verwendung](#9-verwendung)
10. [Fehlerbehandlung](#10-fehlerbehandlung)
11. [Erweiterungsmöglichkeiten](#11-erweiterungsmöglichkeiten)

---

## 1. Projektübersicht

### 1.1 Zielsetzung

Dieses Projekt implementiert einen Web Scraper, der die **IMDb Top 250 Filme** scrapt und mittels eines **lokal laufenden Large Language Models (LLM)** die Filmdaten strukturiert extrahiert. Der innovative Ansatz besteht darin, dass das LLM direkt aus dem bereinigten HTML die relevanten Informationen im JSON-Format extrahiert - ohne klassische XPath- oder CSS-Selektoren.

### 1.2 Kernkonzept: Direkte LLM-Extraktion

Traditionelle Web Scraper verwenden:
- XPath-Ausdrücke
- CSS-Selektoren
- Reguläre Ausdrücke

**Dieser Scraper verwendet stattdessen:**
- Ein lokales LLM (Ollama) zur semantischen Datenextraktion
- Das LLM "versteht" den HTML-Inhalt und extrahiert strukturierte Daten
- Flexibler gegenüber Website-Änderungen
- Keine manuelle Anpassung von Selektoren nötig

### 1.3 Hauptfunktionen

| Funktion | Beschreibung |
|----------|--------------|
| Web Scraping | Playwright-basiertes Scraping mit JavaScript-Rendering |
| HTML-Bereinigung | Optimierte Aufbereitung für LLM-Verarbeitung |
| LLM-Extraktion | Lokale Datenextraktion via Ollama |
| JSON-Export | Strukturierte Ausgabe aller Filmdaten |
| PostgreSQL-Integration | Optionale Speicherung in relationaler Datenbank |
| CLI-Interface | Benutzerfreundliche Kommandozeile mit Fortschrittsanzeige |

---

## 2. Architektur

### 2.1 Systemarchitektur

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           IMDb Top 200 Scraper                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │    IMDb      │    │   Scraper    │    │   Ollama     │    │  Output   │ │
│  │   Website    │───▶│  (Playwright)│───▶│    (LLM)     │───▶│  (JSON/   │ │
│  │              │    │              │    │              │    │   DB)     │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│         │                   │                   │                   │      │
│         ▼                   ▼                   ▼                   ▼      │
│    HTTP/HTTPS          HTML laden          JSON erzeugen      Speichern    │
│                        + bereinigen        aus HTML                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Datenfluss

```
1. SCRAPING
   ┌─────────────┐
   │ IMDb Top250 │──▶ Liste aller Film-URLs extrahieren
   └─────────────┘
         │
         ▼
2. LADEN & BEREINIGEN
   ┌─────────────┐     ┌─────────────┐
   │ Film-Seite  │────▶│ HTML Cleaner│──▶ Bereinigter Text (~8KB)
   │ (Full HTML) │     │             │
   └─────────────┘     └─────────────┘
         │
         ▼
3. LLM-EXTRAKTION
   ┌─────────────┐     ┌─────────────┐
   │ Ollama LLM  │────▶│ JSON Output │──▶ Strukturierte Filmdaten
   │ (qwen3:8b)  │     │             │
   └─────────────┘     └─────────────┘
         │
         ▼
4. SPEICHERUNG
   ┌─────────────┐     ┌─────────────┐
   │ JSON Datei  │     │ PostgreSQL  │
   │ (immer)     │     │ (optional)  │
   └─────────────┘     └─────────────┘
```

### 2.3 Modulstruktur

```
imdb_scraper/
│
├── main.py                      # Einstiegspunkt & CLI
├── config.py                    # Zentrale Konfiguration
│
├── scraper/                     # Web Scraping Modul
│   ├── __init__.py
│   ├── spider.py               # IMDb Scraper (Playwright)
│   └── cleaner.py              # HTML-Bereinigung
│
├── llm/                         # LLM Integration
│   ├── __init__.py
│   ├── ollama_client.py        # Ollama API Client
│   └── prompts.py              # Prompt Templates
│
├── database/                    # Datenbankschicht
│   ├── __init__.py
│   ├── models.py               # SQLAlchemy ORM Models
│   ├── connection.py           # Verbindungsmanagement
│   └── repository.py           # CRUD Operationen
│
├── pipeline/                    # Orchestrierung
│   ├── __init__.py
│   └── processor.py            # Hauptverarbeitungslogik
│
└── output/                      # Ausgabeverzeichnis
    └── *.json                  # Generierte JSON-Dateien
```

---

## 3. Technologie-Stack

### 3.1 Kernkomponenten

| Komponente | Technologie | Version | Zweck |
|------------|-------------|---------|-------|
| Runtime | Python | 3.10+ | Programmiersprache |
| Web Scraping | Playwright | 1.40+ | Browser-Automatisierung mit JS-Rendering |
| HTML Parsing | BeautifulSoup4 | 4.12+ | HTML-Analyse und -Bereinigung |
| LLM Runtime | Ollama | 0.16+ | Lokale LLM-Inferenz |
| LLM Modell | qwen3:8b | - | Open-Source Sprachmodell |
| ORM | SQLAlchemy | 2.0+ | Datenbankabstraktion |
| Datenbank | PostgreSQL | 14+ | Relationale Datenspeicherung |
| CLI | Rich | 13+ | Terminal-UI und Fortschrittsanzeige |

### 3.2 Warum diese Technologien?

**Playwright statt Requests/Scrapy:**
- IMDb lädt Inhalte dynamisch via JavaScript
- Playwright rendert JavaScript vollständig
- Bessere Bot-Erkennung-Umgehung

**Ollama statt Cloud-APIs:**
- Vollständig lokal - keine API-Kosten
- Datenschutz - keine Daten verlassen den Rechner
- Open-Source Modelle

**qwen3:8b als Standardmodell:**
- Gute Balance zwischen Qualität und Geschwindigkeit
- Läuft auf Consumer-Hardware (8GB+ RAM)
- Hervorragende JSON-Generierung

---

## 4. Installation & Setup

### 4.1 Voraussetzungen

```bash
# Systemanforderungen
- macOS 12+, Linux oder Windows 10+
- Python 3.10 oder höher
- 16 GB RAM empfohlen (minimum 8 GB)
- 10 GB freier Speicherplatz
- Internetverbindung für Scraping
```

### 4.2 Automatische Installation

```bash
# Repository klonen/kopieren
cd ~/imdb_scraper

# Setup-Script ausführen
chmod +x setup.sh
./setup.sh
```

Das Setup-Script führt folgende Schritte aus:
1. Python Virtual Environment erstellen
2. Dependencies installieren
3. Playwright Browser (Chromium) installieren
4. `.env` Datei erstellen
5. Ollama und Modell prüfen

### 4.3 Manuelle Installation

```bash
# 1. Virtual Environment
python3 -m venv venv
source venv/bin/activate

# 2. Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Playwright Browser
playwright install chromium

# 4. Konfiguration
cp .env.example .env

# 5. Ollama Modell
ollama pull qwen3:8b
```

### 4.4 Ollama Installation

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Modell laden
ollama pull qwen3:8b

# Server starten (separates Terminal)
ollama serve
```

---

## 5. Konfiguration

### 5.1 Umgebungsvariablen (.env)

```bash
# =============================================================================
# Datenbank (PostgreSQL) - Optional
# =============================================================================
DB_HOST=localhost
DB_PORT=5432
DB_NAME=imdb_movies
DB_USER=postgres
DB_PASSWORD=postgres

# =============================================================================
# Ollama LLM
# =============================================================================
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:8b          # Oder: llama3.1:8b, mistral:7b
OLLAMA_TIMEOUT=120             # Sekunden pro Anfrage

# =============================================================================
# Logging
# =============================================================================
LOG_LEVEL=INFO                 # DEBUG, INFO, WARNING, ERROR
```

### 5.2 Konfigurationsdatei (config.py)

```python
# Scraper-Einstellungen
SCRAPER_CONFIG = {
    "base_url": "https://www.imdb.com",
    "top_250_url": "https://www.imdb.com/chart/top/",
    "max_movies": 200,          # Anzahl zu scrapender Filme
    "delay_seconds": 2,         # Pause zwischen Requests
    "concurrent_requests": 1,   # Sequentielles Scraping
    "user_agent": "Mozilla/5.0 ..."
}

# LLM-Einstellungen
OLLAMA_CONFIG = {
    "host": "http://localhost:11434",
    "model": "qwen3:8b",
    "timeout": 120,
    "options": {
        "temperature": 0.1,     # Niedrig für konsistente Ausgaben
        "num_ctx": 8192,        # Kontext-Fenster
    }
}
```

### 5.3 Verfügbare LLM-Modelle

| Modell | RAM-Bedarf | Geschwindigkeit | Qualität |
|--------|------------|-----------------|----------|
| `qwen3:8b` | ~8 GB | Mittel | Gut |
| `llama3.1:8b` | ~8 GB | Mittel | Gut |
| `mistral:7b` | ~6 GB | Schnell | Gut |
| `llama3.1:70b` | ~48 GB | Langsam | Sehr gut |
| `qwen2.5:14b` | ~12 GB | Mittel | Sehr gut |

---

## 6. Modulbeschreibungen

### 6.1 main.py - Einstiegspunkt

**Zweck:** CLI-Interface und Programmstart

**Funktionen:**
- Argument-Parsing (argparse)
- Konfigurationsanzeige
- Voraussetzungsprüfung
- Pipeline-Start

```python
# Verwendung
python main.py -n 50 --no-db -v

# Argumente
-n, --num-movies    # Anzahl Filme (default: 200)
--no-db             # Ohne PostgreSQL
-v, --verbose       # Debug-Logging
--config            # Nur Konfiguration anzeigen
-y, --yes           # Ohne Bestätigung starten
```

### 6.2 scraper/spider.py - Web Scraper

**Zweck:** IMDb-Seiten laden und Film-URLs extrahieren

**Kernklasse: `IMDbScraper`**

```python
class IMDbScraper:
    def __init__(self, max_movies, delay):
        # Playwright Browser initialisieren
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)

    def get_top_250_urls(self) -> List[Dict]:
        """Extrahiert alle Film-URLs aus der Top 250 Liste."""

    def fetch_movie_detail(self, url: str) -> str:
        """Lädt und bereinigt eine Filmseite."""

    def scrape_movies(self) -> Generator:
        """Generator für alle Filme."""
```

**Warum Playwright?**
- IMDb verwendet JavaScript für dynamische Inhalte
- HTTP 202-Responses bei einfachen Requests
- Playwright rendert die Seite vollständig

### 6.3 scraper/cleaner.py - HTML-Bereinigung

**Zweck:** HTML für LLM-Verarbeitung optimieren

**Kernklasse: `HTMLCleaner`**

```python
class HTMLCleaner:
    # Tags die entfernt werden
    REMOVE_TAGS = ["script", "style", "nav", "footer", "form", ...]

    def clean(self, html: str) -> str:
        """Allgemeine HTML-Bereinigung."""

    def clean_for_imdb(self, html: str) -> str:
        """IMDb-spezifische Extraktion relevanter Bereiche."""
```

**Bereinigungsschritte:**
1. Entfernen von Script/Style-Tags
2. Entfernen von Navigation/Footer
3. Extraktion relevanter Bereiche (Titel, Rating, Plot)
4. Normalisierung von Whitespace
5. Begrenzung auf max. 8000 Zeichen

**Beispiel-Output:**
```
TITEL: The Shawshank Redemption
METADATEN: 1994 | R | 2h 22m
BEWERTUNG: 9.3
GENRES: Drama
BESCHREIBUNG: A banker convicted of uxoricide...
REGISSEUR: Frank Darabont
```

### 6.4 llm/ollama_client.py - LLM-Integration

**Zweck:** Kommunikation mit Ollama API

**Kernklasse: `OllamaClient`**

```python
class OllamaClient:
    def __init__(self):
        self.client = Client(host=OLLAMA_CONFIG["host"])
        self.model = OLLAMA_CONFIG["model"]

    def extract_film_data(self, html_content: str) -> Dict:
        """Extrahiert Filmdaten aus HTML via LLM."""
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options=self.options,
        )
        return self.extract_json(response)

    def is_available(self) -> bool:
        """Prüft ob Ollama Server und Modell verfügbar sind."""
```

**Fehlerbehandlung:**
- JSON-Parsing mit Fallback-Strategien
- Timeout-Handling (120s default)
- `dauer_minuten` erkennt sowohl Minuten-Zahlen als auch "Xh Ym"-Format

### 6.5 llm/prompts.py - Prompt Engineering

**Zweck:** Optimierte Prompts für Datenextraktion

```python
class PromptTemplates:
    @staticmethod
    def get_system_prompt() -> str:
        return """Du bist ein Datenextraktions-Assistent.
        Extrahiere Filminformationen aus dem gegebenen Text.
        Antworte NUR mit validem JSON, ohne Erklärungen."""

    @staticmethod
    def get_extraction_prompt(html: str) -> str:
        return f"""Extrahiere folgende Informationen:
        - titel: Filmtitel
        - original_titel: Originaltitel (falls anders)
        - jahr: Erscheinungsjahr
        - bewertung: IMDb-Bewertung (1-10)
        - anzahl_bewertungen: Anzahl der Bewertungen
        - dauer_minuten: Filmlänge in Minuten
        - regisseur: Name(n) des Regisseurs
        - genres: Liste der Genres
        - beschreibung: Kurze Filmbeschreibung

        HTML-INHALT:
        {html}

        Antworte nur mit JSON:"""
```

**JSON-Schema:**
```json
{
  "titel": "string",
  "original_titel": "string|null",
  "jahr": "integer|null",
  "bewertung": "float",
  "anzahl_bewertungen": "integer|null",
  "dauer_minuten": "integer|null",
  "regisseur": "string",
  "genres": ["string"],
  "beschreibung": "string"
}
```

### 6.6 pipeline/processor.py - Orchestrierung

**Zweck:** Koordination aller Komponenten

**Kernklasse: `MovieProcessor`**

```python
class MovieProcessor:
    def __init__(self, max_movies, use_database):
        self.scraper = IMDbScraper(max_movies)
        self.llm = OllamaClient()
        self.db = DatabaseConnection() if use_database else None

    def run(self) -> List[Dict]:
        """Hauptverarbeitungsschleife."""
        for movie in self.scraper.scrape_movies():
            # 1. HTML bereinigen (bereits im Scraper)
            # 2. LLM-Extraktion
            extracted = self.llm.extract_film_data(movie["html_content"])
            # 3. Speichern
            self.process_movie(movie, session)

    def _save_checkpoint(self, results: List):
        """Speichert Zwischenergebnisse als JSON."""
```

**Features:**
- Fortschrittsanzeige mit Rich
- Checkpoint-Speicherung alle 10 Filme
- Fehlerstatistiken
- Automatischer JSON-Export

### 6.7 database/ - Datenbankschicht

**models.py - ORM-Modelle:**
```python
class Film(Base):
    __tablename__ = "filme"

    id = Column(Integer, primary_key=True)
    imdb_id = Column(String(20), unique=True)
    titel = Column(String(500))
    original_titel = Column(String(500))
    jahr = Column(Integer)
    bewertung = Column(Float)
    anzahl_bewertungen = Column(Integer)
    dauer_minuten = Column(Integer)
    beschreibung = Column(Text)
    regisseur = Column(String(500))
    rang = Column(Integer)
    raw_json = Column(JSONB)

    genres = relationship("Genre", secondary=film_genres)

class Genre(Base):
    __tablename__ = "genres"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True)
```

**repository.py - CRUD-Operationen:**
```python
class FilmRepository:
    def upsert_film(self, film_data: Dict) -> Film:
        """Erstellt oder aktualisiert einen Film (Insert or Update)."""

    def get_by_imdb_id(self, imdb_id: str) -> Film:
        """Sucht Film nach IMDb-ID."""

    def get_all(self, limit=None, offset=0) -> List[Film]:
        """Alle Filme abrufen."""
```

---

## 7. LLM-basierte Datenextraktion

### 7.1 Konzept

```
┌─────────────────────────────────────────────────────────────────┐
│                    Traditioneller Ansatz                        │
├─────────────────────────────────────────────────────────────────┤
│  HTML ──▶ XPath/CSS-Selektoren ──▶ Werte extrahieren           │
│                                                                 │
│  Problem: Selektoren brechen bei Website-Änderungen            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    LLM-basierter Ansatz                         │
├─────────────────────────────────────────────────────────────────┤
│  HTML ──▶ Bereinigung ──▶ LLM ──▶ Strukturiertes JSON          │
│                                                                 │
│  Vorteil: LLM "versteht" semantisch, robust gegen Änderungen   │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Prompt-Strategie

**System-Prompt:**
- Klare Rollendefiniton (Datenextraktions-Assistent)
- Strikte JSON-only Ausgabe
- Keine Erklärungen oder Kommentare

**User-Prompt:**
- Explizites Schema mit Feldnamen und Typen
- Bereinigter HTML-Inhalt
- Beispiele für erwartete Ausgabe (Few-Shot)

### 7.3 JSON-Parsing

```python
def extract_json(self, text: str) -> Optional[Dict]:
    # Strategie 1: Direktes Parsing
    try:
        return json.loads(text.strip())
    except JSONDecodeError:
        pass

    # Strategie 2: JSON aus Markdown extrahieren (```json ... ```)
    # Strategie 3: Allgemeiner Code-Block (``` ... ```)
    # Strategie 4: Einfaches JSON-Objekt { ... }
    # Strategie 5: Beliebiges JSON (greedy)
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
            except JSONDecodeError:
                continue

    return None  # Kein valides JSON gefunden
```

### 7.4 Optimierungen

| Optimierung | Beschreibung |
|-------------|--------------|
| Niedriges Temperature | `0.1` für konsistente, deterministische Ausgaben |
| Großes Kontextfenster | `8192` Token für vollständige HTML-Verarbeitung |
| HTML-Bereinigung | Reduziert Token-Verbrauch um ~90% |
| Strukturierter Output | Explizites JSON-Schema im Prompt |

---

## 8. Datenbank-Schema

### 8.1 ER-Diagramm

```
┌─────────────────────┐       ┌─────────────────────┐
│       filme         │       │       genres        │
├─────────────────────┤       ├─────────────────────┤
│ id (PK)             │       │ id (PK)             │
│ imdb_id (UNIQUE)    │       │ name (UNIQUE)       │
│ titel               │       └─────────────────────┘
│ original_titel      │                 │
│ jahr                │                 │
│ bewertung           │                 │
│ anzahl_bewertungen  │       ┌─────────────────────┐
│ dauer_minuten       │       │    film_genres      │
│ beschreibung        │       ├─────────────────────┤
│ regisseur           │       │ film_id (FK)        │
│ poster_url          │◀──────│ genre_id (FK)       │
│ imdb_url            │       └─────────────────────┘
│ rang                │
│ raw_json (JSON)     │
│ created_at          │
│ updated_at          │
└─────────────────────┘
```

### 8.2 SQL-Schema

```sql
-- Filme-Tabelle
CREATE TABLE filme (
    id SERIAL PRIMARY KEY,
    imdb_id VARCHAR(20) UNIQUE NOT NULL,
    titel VARCHAR(500) NOT NULL,
    original_titel VARCHAR(500),
    jahr INTEGER,
    bewertung FLOAT,
    anzahl_bewertungen INTEGER,
    dauer_minuten INTEGER,
    beschreibung TEXT,
    regisseur VARCHAR(500),
    poster_url VARCHAR(1000),
    imdb_url VARCHAR(500),
    rang INTEGER,
    raw_json JSON,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Genres-Tabelle
CREATE TABLE genres (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL
);

-- Verknüpfungstabelle (n:m)
CREATE TABLE film_genres (
    film_id INTEGER REFERENCES filme(id) ON DELETE CASCADE,
    genre_id INTEGER REFERENCES genres(id) ON DELETE CASCADE,
    PRIMARY KEY (film_id, genre_id)
);

-- Indizes für Performance
CREATE INDEX idx_filme_jahr ON filme(jahr);
CREATE INDEX idx_filme_bewertung ON filme(bewertung);
CREATE INDEX idx_filme_rang ON filme(rang);
```

### 8.3 PostgreSQL Setup

```bash
# Datenbank erstellen
createdb imdb_movies

# Oder via psql
psql -U postgres -c "CREATE DATABASE imdb_movies;"

# Tabellen werden automatisch erstellt (SQLAlchemy)
```

---

## 9. Verwendung

### 9.1 Grundlegende Befehle

```bash
# Virtual Environment aktivieren
source venv/bin/activate

# Ollama starten (separates Terminal)
ollama serve

# Test mit 10 Filmen (ohne Datenbank)
python main.py -n 10 --no-db

# Vollständiger Durchlauf (200 Filme)
python main.py -n 200 -y

# Mit Datenbank
python main.py -n 200 -y

# Verbose-Modus für Debugging
python main.py -n 5 --no-db -v
```

### 9.2 CLI-Optionen

| Option | Kurzform | Beschreibung | Default |
|--------|----------|--------------|---------|
| `--num-movies` | `-n` | Anzahl zu scrapender Filme | 200 |
| `--no-db` | - | Ohne PostgreSQL (nur JSON) | False |
| `--verbose` | `-v` | Debug-Ausgaben | False |
| `--config` | - | Nur Konfiguration anzeigen | - |
| `--yes` | `-y` | Ohne Bestätigung starten | False |

### 9.3 Output

**JSON-Datei (immer erzeugt):**
```
output/imdb_top_200_20260318_172635.json
```

**Struktur:**
```json
{
  "metadata": {
    "scraped_at": "2026-03-18T17:26:35",
    "total_movies": 200,
    "llm_model": "qwen3:8b"
  },
  "movies": [
    {
      "imdb_id": "tt0111161",
      "rang": 1,
      "titel": "The Shawshank Redemption",
      "jahr": 1994,
      "bewertung": 9.3,
      "regisseur": "Frank Darabont",
      "genres": ["Drama"],
      "beschreibung": "..."
    }
  ]
}
```

---

## 10. Fehlerbehandlung

### 10.1 Häufige Fehler

| Fehler | Ursache | Lösung |
|--------|---------|--------|
| "Ollama nicht erreichbar" | Server nicht gestartet | `ollama serve` ausführen |
| "Modell nicht gefunden" | Modell nicht installiert | `ollama pull qwen3:8b` |
| "Timeout bei LLM" | Modell zu langsam | Kleineres Modell oder Timeout erhöhen |
| "Sehr wenig Inhalt" | JavaScript nicht gerendert | Playwright wird automatisch verwendet |
| "PostgreSQL Verbindung fehlgeschlagen" | DB nicht erreichbar | `--no-db` verwenden oder DB starten |

### 10.2 Debugging

```bash
# Verbose-Modus aktivieren
python main.py -n 3 --no-db -v

# Ollama-Status prüfen
curl http://localhost:11434/api/tags

# Modelle auflisten
ollama list

# PostgreSQL-Verbindung testen
psql -h localhost -U postgres -d imdb_movies -c "SELECT 1;"
```

### 10.3 Logs

```
2026-03-18 17:28:01 - scraper.spider - INFO - [1/200] Scrape: The Shawshank Redemption
2026-03-18 17:29:19 - llm.ollama_client - INFO - Film extrahiert: The Shawshank Redemption
```

---

## 11. Erweiterungsmöglichkeiten

### 11.1 Weitere Datenquellen

- Rotten Tomatoes Scores integrieren
- Trailer-URLs von YouTube
- Box Office Daten
- Awards/Nominierungen

### 11.2 Performance-Optimierungen

- Parallele LLM-Anfragen (mehrere Ollama-Instanzen)
- Caching von bereits gescrapten Filmen
- Inkrementelle Updates (nur neue Filme)

### 11.3 Alternative LLM-Modelle

```bash
# Schnelleres Modell
ollama pull mistral:7b
# In .env: OLLAMA_MODEL=mistral:7b

# Höhere Qualität
ollama pull llama3.1:70b
# In .env: OLLAMA_MODEL=llama3.1:70b
```

### 11.4 API-Erweiterung

```python
# FastAPI-Endpunkte für Abfragen
from fastapi import FastAPI

app = FastAPI()

@app.get("/movies")
def get_movies():
    return repository.get_all()

@app.get("/movies/{imdb_id}")
def get_movie(imdb_id: str):
    return repository.get_by_imdb_id(imdb_id)
```

---

## Anhang

### A. Abhängigkeiten (requirements.txt)

```
# Web Scraping
playwright>=1.40.0
beautifulsoup4>=4.12.0
lxml>=5.0.0

# LLM
ollama>=0.4.0

# Database
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0

# Utilities
python-dotenv>=1.0.0
rich>=13.7.0
```

### B. Beispiel-Ausgabe

```json
{
  "imdb_id": "tt0111161",
  "imdb_url": "https://www.imdb.com/title/tt0111161/",
  "rang": 1,
  "titel": "The Shawshank Redemption",
  "original_titel": null,
  "jahr": 1994,
  "bewertung": 9.3,
  "anzahl_bewertungen": 2800000,
  "dauer_minuten": 142,
  "regisseur": "Frank Darabont",
  "genres": ["Drama"],
  "beschreibung": "A banker convicted of uxoricide forms a friendship over a quarter century with a hardened convict, while maintaining his innocence and trying to remain hopeful through simple compassion."
}
```

### C. Lizenz

MIT License

---

## Changelog

### Version 1.2 (2026-03-23)

**Bugfixes:**

| # | Datei | Problem | Fix |
|---|-------|---------|-----|
| 1 | `database/repository.py` | `update_film()`: Loop `setattr(film, "genres", [...])` versuchte SQLAlchemy-Relationship auf Strings zu setzen → Fehler beim Flush | `"genres"` im Felder-Loop übersprungen; Genres werden weiterhin korrekt im separaten Block behandelt |
| 2 | `evaluation/evaluate.py` | Kommentar sagte `±0.1`, Code verwendete `0.15` | Kommentar auf `±0.15` korrigiert |
| 3 | `pipeline/processor.py` | Im `--no-db` Modus wurden keine Checkpoints gespeichert → bei Absturz Datenverlust | Checkpoint alle 10 Filme auch im Test-Modus |

**Aufräumen:**

- `DOKUMENTATION.md`: Pydantic aus der Tech-Stack-Tabelle entfernt (Paket nicht im Code verwendet)

---

### Version 1.1 (2026-03-22)

**Bugfixes:**

| # | Datei | Problem | Fix |
|---|-------|---------|-----|
| 1 | `llm/ollama_client.py` | `dauer_minuten`: "2h 22m" wurde als `2` statt `142` Minuten extrahiert | Stunden/Minuten-Format wird jetzt korrekt in Gesamtminuten umgerechnet |
| 2 | `database/repository.py` | `film_data.pop("genres")` mutierte das Input-Dict → Genres fehlten im JSON-Output | `.pop()` durch `.get()` ersetzt |
| 3 | `database/repository.py` | `session.rollback()` in `create_film()` rollte die gesamte Session zurück (bis zu 10 Filme verloren bei einem Duplikat-Fehler) | Savepoint via `session.begin_nested()` isoliert den Rollback auf den einzelnen Film |
| 4 | `database/models.py` | `datetime.utcnow` ist seit Python 3.12 deprecated | Ersetzt durch `lambda: datetime.now(timezone.utc)` |

**Aufräumen:**

- `pipeline/processor.py`: Doppelte `main()`-Funktion entfernt (toter Code)
- `scraper/cleaner.py`: Nie benutztes `REMOVE_ATTRIBUTES` Class-Attribut entfernt
- `requirements.txt`: Nicht verwendete Pakete entfernt (`scrapy`, `scrapy-playwright`, `alembic`, `pydantic`)
