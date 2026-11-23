#!/bin/bash
# =============================================================================
# IMDb Scraper Setup Script
# =============================================================================

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           IMDb Scraper - Setup                               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Farben
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# 1. Python Virtual Environment
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[1/5] Erstelle Python Virtual Environment...${NC}"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual Environment erstellt${NC}"
else
    echo -e "${GREEN}✓ Virtual Environment existiert bereits${NC}"
fi

source venv/bin/activate

# -----------------------------------------------------------------------------
# 2. Python Dependencies
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[2/5] Installiere Python Dependencies...${NC}"

pip install --upgrade pip
pip install -r requirements.txt

echo -e "${GREEN}✓ Dependencies installiert${NC}"

# -----------------------------------------------------------------------------
# 3. Playwright Browser (für dynamische Seiten)
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[3/5] Installiere Playwright Browser...${NC}"

playwright install chromium

echo -e "${GREEN}✓ Playwright Browser installiert${NC}"

# -----------------------------------------------------------------------------
# 4. .env Datei
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[4/5] Konfiguration...${NC}"

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ .env Datei erstellt (bitte anpassen)${NC}"
else
    echo -e "${GREEN}✓ .env Datei existiert bereits${NC}"
fi

# Output Verzeichnis erstellen
mkdir -p output

# -----------------------------------------------------------------------------
# 5. Ollama prüfen
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[5/5] Prüfe Ollama...${NC}"

if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama ist installiert${NC}"

    # Prüfen ob Ollama läuft
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama Server läuft${NC}"

        # Modell prüfen/laden
        if ollama list | grep -q "llama3.1:8b"; then
            echo -e "${GREEN}✓ Modell llama3.1:8b ist vorhanden${NC}"
        else
            echo -e "${YELLOW}Lade Modell llama3.1:8b...${NC}"
            ollama pull llama3.1:8b
            echo -e "${GREEN}✓ Modell geladen${NC}"
        fi
    else
        echo -e "${RED}✗ Ollama Server läuft nicht${NC}"
        echo -e "${YELLOW}  Starte mit: ollama serve${NC}"
    fi
else
    echo -e "${RED}✗ Ollama ist nicht installiert${NC}"
    echo -e "${YELLOW}  Installiere von: https://ollama.ai${NC}"
fi

# -----------------------------------------------------------------------------
# Zusammenfassung
# -----------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           Setup abgeschlossen!                               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Nächste Schritte:"
echo ""
echo "  1. Ollama starten (falls nicht laufend):"
echo "     ollama serve"
echo ""
echo "  2. PostgreSQL Datenbank erstellen (optional):"
echo "     createdb imdb_movies"
echo ""
echo "  3. .env Datei anpassen (falls nötig):"
echo "     nano .env"
echo ""
echo "  4. Scraper starten:"
echo "     source venv/bin/activate"
echo "     python main.py -n 10 --no-db    # Test: 10 Filme, ohne DB"
echo "     python main.py                   # Vollständig: 200 Filme"
echo ""
