#!/usr/bin/env python3
"""
IMDb Top 250 Scraper mit LLM-basierter Datenextraktion

Dieses Skript scrapt die IMDb Top 250 Liste, extrahiert Filminformationen
mittels eines lokalen LLM (Ollama) und speichert die Daten in PostgreSQL.

Verwendung:
    python main.py                    # Standard: 250 Filme (volle Top 250)
    python main.py -n 50              # Nur 50 Filme
    python main.py --no-db            # Ohne Datenbank (Test)
    python main.py -v                 # Verbose Logging

Voraussetzungen:
    1. Ollama muss laufen: `ollama serve`
    2. Modell muss vorhanden sein: `ollama pull qwen2.5:3b`
    3. PostgreSQL muss laufen (optional)
"""
import sys
import logging
from pathlib import Path

# Projekt-Root zum Path hinzufügen
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from config import OLLAMA_CONFIG, DATABASE_CONFIG, SCRAPER_CONFIG
from pipeline import MovieProcessor

console = Console()


def print_banner():
    """Zeigt das Start-Banner an."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   ██╗███╗   ███╗██████╗ ██████╗                              ║
    ║   ██║████╗ ████║██╔══██╗██╔══██╗                             ║
    ║   ██║██╔████╔██║██║  ██║██████╔╝                             ║
    ║   ██║██║╚██╔╝██║██║  ██║██╔══██╗                             ║
    ║   ██║██║ ╚═╝ ██║██████╔╝██████╔╝                             ║
    ║   ╚═╝╚═╝     ╚═╝╚═════╝ ╚═════╝                              ║
    ║                                                              ║
    ║   Top 250 Scraper mit LLM-Extraktion                         ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def print_config():
    """Zeigt die aktuelle Konfiguration an."""
    config_info = f"""
[bold]Konfiguration:[/bold]

[cyan]LLM (Ollama):[/cyan]
  • Host: {OLLAMA_CONFIG['host']}
  • Modell: {OLLAMA_CONFIG['model']}
  • Timeout: {OLLAMA_CONFIG['timeout']}s

[cyan]Datenbank (PostgreSQL):[/cyan]
  • Host: {DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}
  • Database: {DATABASE_CONFIG['database']}
  • User: {DATABASE_CONFIG['user']}

[cyan]Scraper:[/cyan]
  • Max Filme: {SCRAPER_CONFIG['max_movies']}
  • Delay: {SCRAPER_CONFIG['delay_seconds']}s
"""
    console.print(Panel(config_info, title="Konfiguration", border_style="blue"))


def main():
    """Hauptfunktion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="IMDb Top 250 Scraper mit LLM-basierter Datenextraktion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python main.py                    # Standard: 250 Filme (volle Top 250)
  python main.py -n 50              # Nur 50 Filme
  python main.py --no-db            # Ohne Datenbank (Test)
  python main.py -n 10 --no-db -v   # 10 Filme, Test-Modus, Verbose

Voraussetzungen:
  1. Ollama installieren: https://ollama.ai
  2. Ollama starten: ollama serve
  3. Modell laden: ollama pull qwen2.5:3b
  4. (Optional) PostgreSQL einrichten
        """
    )

    parser.add_argument(
        "-n", "--num-movies",
        type=int,
        default=250,
        help="Anzahl der zu scrapenden Filme (default: 250)"
    )

    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Ohne Datenbank ausführen (Test-Modus, speichert nur JSON)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Ausführliche Logging-Ausgabe"
    )

    parser.add_argument(
        "--config",
        action="store_true",
        help="Zeigt nur die Konfiguration an"
    )

    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Keine Bestätigung vor dem Start"
    )

    args = parser.parse_args()

    # Logging konfigurieren
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(project_root / "output" / "scraper.log"),
        ]
    )

    # Banner anzeigen
    print_banner()

    # Nur Konfiguration anzeigen
    if args.config:
        print_config()
        return 0

    # Konfiguration anzeigen
    print_config()

    # Bestätigung
    if not args.yes:
        console.print()
        if not Confirm.ask(
            f"[bold]Starte Scraping von {args.num_movies} Filmen?[/bold]"
        ):
            console.print("[yellow]Abgebrochen.[/yellow]")
            return 0

    # Processor starten
    try:
        processor = MovieProcessor(
            max_movies=args.num_movies,
            use_database=not args.no_db
        )

        movies = processor.run()

        if movies:
            console.print(
                f"\n[bold green]✓ Erfolgreich {len(movies)} Filme verarbeitet![/bold green]"
            )
            return 0
        else:
            console.print(
                "\n[bold red]✗ Keine Filme verarbeitet.[/bold red]"
            )
            return 1

    except KeyboardInterrupt:
        console.print("\n[yellow]Abgebrochen durch Benutzer.[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[bold red]Fehler: {e}[/bold red]")
        logging.exception("Unerwarteter Fehler")
        return 1


if __name__ == "__main__":
    sys.exit(main())
