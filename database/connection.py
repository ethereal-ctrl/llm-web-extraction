"""
Datenbank-Verbindung und Session-Management
"""
import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from config import DATABASE_URL
from database.models import Base

logger = logging.getLogger(__name__)


def get_engine(database_url: str = DATABASE_URL, echo: bool = False):
    """
    Erstellt eine SQLAlchemy Engine.

    Args:
        database_url: PostgreSQL Verbindungs-URL
        echo: SQL-Logging aktivieren

    Returns:
        SQLAlchemy Engine
    """
    return create_engine(
        database_url,
        echo=echo,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,  # Verbindung vor Nutzung prüfen
    )


def get_session(engine) -> sessionmaker:
    """
    Erstellt eine Session-Factory.

    Args:
        engine: SQLAlchemy Engine

    Returns:
        Session-Factory
    """
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


class DatabaseConnection:
    """
    Datenbank-Verbindungsmanager mit Context-Manager Support.
    """

    def __init__(self, database_url: str = DATABASE_URL, echo: bool = False):
        """
        Initialisiert die Datenbankverbindung.

        Args:
            database_url: PostgreSQL Verbindungs-URL
            echo: SQL-Logging aktivieren
        """
        self.database_url = database_url
        self.engine = get_engine(database_url, echo)
        self.SessionLocal = get_session(self.engine)

    def create_tables(self) -> None:
        """Erstellt alle Tabellen in der Datenbank."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Datenbank-Tabellen erfolgreich erstellt.")
        except SQLAlchemyError as e:
            logger.error(f"Fehler beim Erstellen der Tabellen: {e}")
            raise

    def drop_tables(self) -> None:
        """Löscht alle Tabellen (Vorsicht!)."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("Alle Datenbank-Tabellen wurden gelöscht.")
        except SQLAlchemyError as e:
            logger.error(f"Fehler beim Löschen der Tabellen: {e}")
            raise

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Context-Manager für Datenbank-Sessions.

        Yields:
            SQLAlchemy Session

        Example:
            with db.session() as session:
                session.add(film)
                session.commit()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Datenbank-Fehler: {e}")
            raise
        finally:
            session.close()

    def test_connection(self) -> bool:
        """
        Testet die Datenbankverbindung.

        Returns:
            True wenn Verbindung erfolgreich
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Datenbankverbindung erfolgreich.")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Datenbankverbindung fehlgeschlagen: {e}")
            return False

    def close(self) -> None:
        """Schließt alle Verbindungen."""
        self.engine.dispose()
        logger.info("Datenbankverbindungen geschlossen.")
