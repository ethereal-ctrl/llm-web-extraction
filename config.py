"""
Konfiguration für den IMDb Scraper
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# .env Datei laden
load_dotenv()

# Basis-Pfad
BASE_DIR = Path(__file__).parent

# =============================================================================
# Datenbank Konfiguration
# =============================================================================
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "imdb_movies"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

DATABASE_URL = (
    f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}"
    f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
)

# =============================================================================
# Ollama LLM Konfiguration
# =============================================================================
OLLAMA_CONFIG = {
    "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    "model": os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),  # Lokal, Open-Source — finale Modellwahl der Arbeit
    "timeout": int(os.getenv("OLLAMA_TIMEOUT", 120)),
    "options": {
        "temperature": 0.1,  # Niedrig für konsistente Ausgaben
        "num_ctx": 8192,     # Kontext-Fenster
    }
}

# =============================================================================
# Scraper Konfiguration
# =============================================================================
SCRAPER_CONFIG = {
    "base_url": "https://www.imdb.com",
    "top_250_url": "https://www.imdb.com/chart/top/",
    "max_movies": 250,  # Top 250 Filme (vollständige IMDb-Top-250-Liste)
    "delay_seconds": 2,  # Verzögerung zwischen Requests
    "concurrent_requests": 1,  # Sequentiell für Rate-Limiting
    "user_agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# =============================================================================
# Logging Konfiguration
# =============================================================================
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}

# =============================================================================
# Ausgabe Konfiguration
# =============================================================================
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
