"""
Movie Processor - Orchestriert Scraping, LLM-Extraktion und Speicherung
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from config import OUTPUT_DIR
from scraper import IMDbScraper
from llm import OllamaClient
from database import DatabaseConnection, FilmRepository

logger = logging.getLogger(__name__)
console = Console()


class MovieProcessor:
    """
    Orchestriert den gesamten Scraping- und Extraktionsprozess.

    1. Scrapt IMDb Top 250
    2. Extrahiert Daten via LLM
    3. Speichert in PostgreSQL
    4. Exportiert als JSON
    """

    def __init__(
        self,
        max_movies: int = 200,
        use_database: bool = True,
    ):
        """
        Initialisiert den Processor.

        Args:
            max_movies: Maximale Anzahl zu verarbeitender Filme
            use_database: Datenbank verwenden (False für Test-Modus)
        """
        self.max_movies = max_movies
        self.use_database = use_database

        # Komponenten initialisieren
        self.scraper = IMDbScraper(max_movies=max_movies)
        self.llm = OllamaClient()

        if use_database:
            self.db = DatabaseConnection()
            self.db.create_tables()
        else:
            self.db = None

        # Statistiken
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "start_time": None,
            "end_time": None,
        }

        # Resume: lade bereits gescrapte IDs aus Checkpoint
        self._checkpoint_file = OUTPUT_DIR / "checkpoint.json"
        self._resumed_movies, self._scraped_ids = self._load_checkpoint()

        logger.info(f"MovieProcessor initialisiert: max_movies={max_movies}")

    def check_prerequisites(self) -> bool:
        """
        Prüft ob alle Voraussetzungen erfüllt sind.

        Returns:
            True wenn alles bereit
        """
        console.print("\n[bold blue]Prüfe Voraussetzungen...[/bold blue]\n")

        checks = []

        # Ollama verfügbar?
        ollama_ok = self.llm.is_available()
        checks.append(("Ollama LLM", ollama_ok))

        if not ollama_ok:
            console.print(
                "[yellow]Hinweis: Modell wird heruntergeladen...[/yellow]"
            )
            self.llm.pull_model()
            ollama_ok = self.llm.is_available()
            checks[-1] = ("Ollama LLM", ollama_ok)

        # Datenbank verfügbar?
        if self.use_database:
            db_ok = self.db.test_connection()
            checks.append(("PostgreSQL", db_ok))
        else:
            checks.append(("PostgreSQL", "Übersprungen (Test-Modus)"))

        # Ergebnisse anzeigen
        table = Table(title="Voraussetzungen")
        table.add_column("Komponente", style="cyan")
        table.add_column("Status", style="green")

        all_ok = True
        for name, status in checks:
            if isinstance(status, bool):
                status_text = "✓ OK" if status else "✗ Fehler"
                style = "green" if status else "red"
                if not status:
                    all_ok = False
            else:
                status_text = status
                style = "yellow"

            table.add_row(name, f"[{style}]{status_text}[/{style}]")

        console.print(table)

        return all_ok

    def process_movie(
        self,
        movie_data: Dict[str, Any],
        session=None,
    ) -> Optional[Dict[str, Any]]:
        """
        Verarbeitet einen einzelnen Film.

        Args:
            movie_data: Daten aus dem Scraper
            session: Datenbank-Session

        Returns:
            Verarbeitete Film-Daten oder None
        """
        html_content = movie_data.get("html_content")

        if not html_content:
            logger.warning(f"Kein HTML für {movie_data.get('titel')}")
            return None

        # LLM-Extraktion
        extracted = self.llm.extract_film_data(html_content)

        if not extracted:
            logger.warning(f"LLM-Extraktion fehlgeschlagen für {movie_data.get('titel')}")
            return None

        # Daten zusammenführen
        film_data = {
            "imdb_id": movie_data.get("imdb_id"),
            "imdb_url": movie_data.get("url"),
            "rang": movie_data.get("rang"),
            "titel": extracted.get("titel") or movie_data.get("titel"),
            "original_titel": extracted.get("original_titel"),
            "jahr": extracted.get("jahr") or movie_data.get("jahr"),
            "bewertung": extracted.get("bewertung") or movie_data.get("bewertung"),
            "anzahl_bewertungen": extracted.get("anzahl_bewertungen"),
            "dauer_minuten": extracted.get("dauer_minuten"),
            "regisseur": extracted.get("regisseur"),
            "genres": extracted.get("genres", []),
            "beschreibung": extracted.get("beschreibung"),
            "raw_json": extracted,  # Original LLM-Output speichern
        }

        # In Datenbank speichern
        if session:
            repo = FilmRepository(session)
            film = repo.upsert_film(film_data)
            if film:
                logger.info(f"Film gespeichert: {film.titel} (ID: {film.id})")

        return film_data

    def run(self) -> List[Dict[str, Any]]:
        """
        Führt den kompletten Scraping-Prozess aus.

        Returns:
            Liste aller verarbeiteten Filme
        """
        # Voraussetzungen prüfen
        if not self.check_prerequisites():
            console.print(
                "\n[bold red]Voraussetzungen nicht erfüllt. Abbruch.[/bold red]"
            )
            return []

        console.print(
            Panel(
                f"[bold green]Starte IMDb Top {self.max_movies} Scraping[/bold green]\n"
                f"LLM: {self.llm.model}\n"
                f"Datenbank: {'Aktiv' if self.use_database else 'Deaktiviert'}",
                title="IMDb Scraper"
            )
        )

        self.stats["start_time"] = datetime.now()
        processed_movies = []

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:

                # Scraping Task
                task = progress.add_task(
                    "[cyan]Scrape & Extrahiere Filme...",
                    total=self.max_movies
                )

                if self.use_database:
                    with self.db.session() as session:
                        # Resume: bereits gescrapte Filme vorladen
                        if self._resumed_movies:
                            processed_movies.extend(self._resumed_movies)
                            self.stats["success"] += len(self._resumed_movies)
                            self.stats["total"] += len(self._resumed_movies)
                            progress.update(task, advance=len(self._resumed_movies))
                            console.print(
                                f"[yellow]Resume: {len(self._resumed_movies)} Filme "
                                f"aus Checkpoint übersprungen[/yellow]"
                            )

                        for movie_data in self.scraper.scrape_movies():
                            if movie_data.get("imdb_id") in self._scraped_ids:
                                continue

                            self.stats["total"] += 1

                            result = self.process_movie(movie_data, session)

                            if result:
                                processed_movies.append(result)
                                self.stats["success"] += 1
                            else:
                                self.stats["failed"] += 1

                            progress.update(task, advance=1)

                            # Zwischenspeichern alle 10 Filme
                            if self.stats["total"] % 10 == 0:
                                session.commit()
                                self._save_checkpoint(processed_movies)
                else:
                    # Test-Modus ohne Datenbank
                    # Resume: bereits gescrapte Filme vorladen
                    if self._resumed_movies:
                        processed_movies.extend(self._resumed_movies)
                        self.stats["success"] += len(self._resumed_movies)
                        self.stats["total"] += len(self._resumed_movies)
                        progress.update(task, advance=len(self._resumed_movies))
                        console.print(
                            f"[yellow]Resume: {len(self._resumed_movies)} Filme "
                            f"aus Checkpoint übersprungen[/yellow]"
                        )

                    for movie_data in self.scraper.scrape_movies():
                        # Bereits gescrapt? Überspringen.
                        if movie_data.get("imdb_id") in self._scraped_ids:
                            continue

                        self.stats["total"] += 1

                        result = self.process_movie(movie_data)

                        if result:
                            processed_movies.append(result)
                            self.stats["success"] += 1
                        else:
                            self.stats["failed"] += 1

                        progress.update(task, advance=1)

                        # Zwischenspeichern alle 10 Filme
                        if self.stats["total"] % 10 == 0:
                            self._save_checkpoint(processed_movies)

        except KeyboardInterrupt:
            console.print("\n[yellow]Abbruch durch Benutzer — Checkpoint wird gespeichert...[/yellow]")
            self._save_checkpoint(processed_movies)
            console.print(f"[green]Checkpoint gespeichert: {len(processed_movies)} Filme. Beim nächsten Start wird fortgesetzt.[/green]")
        except Exception as e:
            logger.error(f"Fehler während der Verarbeitung: {e}")
            console.print(f"\n[red]Fehler: {e}[/red]")
            self._save_checkpoint(processed_movies)
        finally:
            self.stats["end_time"] = datetime.now()
            self.scraper.close()

        # Ergebnisse speichern und anzeigen
        self._save_results(processed_movies)
        self._print_summary()

        # Checkpoint löschen nach erfolgreichem Abschluss
        if self._checkpoint_file.exists() and self.stats["failed"] == 0:
            self._checkpoint_file.unlink()
            logger.debug("Checkpoint gelöscht nach erfolgreichem Abschluss")

        return processed_movies

    def _load_checkpoint(self):
        """Lädt Checkpoint-Daten für Resume-Funktionalität."""
        if not self._checkpoint_file.exists():
            return [], set()
        try:
            with open(self._checkpoint_file, "r", encoding="utf-8") as f:
                movies = json.load(f)
            ids = {m["imdb_id"] for m in movies if m.get("imdb_id")}
            if ids:
                logger.info(f"Resume: {len(ids)} Filme aus Checkpoint geladen")
            return movies, ids
        except Exception as e:
            logger.warning(f"Checkpoint laden fehlgeschlagen: {e}")
            return [], set()

    def _save_checkpoint(self, movies: List[Dict[str, Any]]) -> None:
        """Speichert Zwischenstand."""
        try:
            with open(self._checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(movies, f, ensure_ascii=False, indent=2, default=str)
            logger.debug(f"Checkpoint gespeichert: {len(movies)} Filme")
        except Exception as e:
            logger.warning(f"Checkpoint speichern fehlgeschlagen: {e}")

    def _save_results(self, movies: List[Dict[str, Any]]) -> None:
        """Speichert finale Ergebnisse als JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Vollständige Ergebnisse
        output_file = OUTPUT_DIR / f"imdb_top_{self.max_movies}_{timestamp}.json"

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metadata": {
                            "scraped_at": datetime.now().isoformat(),
                            "total_movies": len(movies),
                            "llm_model": self.llm.model,
                        },
                        "movies": movies,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=str
                )

            console.print(f"\n[green]Ergebnisse gespeichert: {output_file}[/green]")

        except Exception as e:
            logger.error(f"Speichern fehlgeschlagen: {e}")

    def _print_summary(self) -> None:
        """Zeigt Zusammenfassung an."""
        duration = None
        if self.stats["start_time"] and self.stats["end_time"]:
            duration = self.stats["end_time"] - self.stats["start_time"]

        table = Table(title="\nZusammenfassung")
        table.add_column("Metrik", style="cyan")
        table.add_column("Wert", style="green")

        table.add_row("Gesamt verarbeitet", str(self.stats["total"]))
        table.add_row("Erfolgreich", f"[green]{self.stats['success']}[/green]")
        table.add_row("Fehlgeschlagen", f"[red]{self.stats['failed']}[/red]")

        if duration:
            table.add_row("Dauer", str(duration).split(".")[0])
            if self.stats["total"] > 0:
                avg = duration.total_seconds() / self.stats["total"]
                table.add_row("Durchschnitt/Film", f"{avg:.1f}s")

        success_rate = 0
        if self.stats["total"] > 0:
            success_rate = (self.stats["success"] / self.stats["total"]) * 100
        table.add_row("Erfolgsrate", f"{success_rate:.1f}%")

        console.print(table)


