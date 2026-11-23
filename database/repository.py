"""
Repository-Pattern für Datenbankoperationen
"""
import logging
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select, func

from database.models import Film, Genre, Person, FilmPerson, film_genres

logger = logging.getLogger(__name__)


class FilmRepository:
    """
    Repository für Film-CRUD-Operationen.
    """

    def __init__(self, session: Session):
        """
        Initialisiert das Repository.

        Args:
            session: SQLAlchemy Session
        """
        self.session = session

    def get_or_create_genre(self, name: str) -> Genre:
        """
        Holt oder erstellt ein Genre.

        Args:
            name: Genre-Name

        Returns:
            Genre-Objekt
        """
        genre = self.session.query(Genre).filter(Genre.name == name).first()
        if not genre:
            genre = Genre(name=name)
            self.session.add(genre)
            self.session.flush()  # ID generieren
        return genre

    def get_or_create_person(self, name: str) -> Person:
        """Findet eine Person über den Namen oder legt sie neu an."""
        person = self.session.query(Person).filter(Person.name == name).first()
        if not person:
            person = Person(name=name)
            self.session.add(person)
            self.session.flush()  # ID generieren
        return person

    def _set_besetzung(self, film: Film, besetzung) -> None:
        """Schreibt die m:n-Besetzung (Schauspieler + Rolle) eines Films."""
        film.besetzung.clear()
        for idx, entry in enumerate(besetzung or []):
            if not isinstance(entry, dict):
                continue
            name = (entry.get("name") or "").strip()
            if not name:
                continue
            person = self.get_or_create_person(name)
            rolle = entry.get("als_charakter")
            film.besetzung.append(
                FilmPerson(
                    person=person,
                    als_charakter=(rolle.strip() if rolle else None),
                    ordnung=idx,
                )
            )

    def create_film(self, film_data: Dict[str, Any]) -> Optional[Film]:
        """
        Erstellt einen neuen Film in der Datenbank.

        Args:
            film_data: Dictionary mit Film-Daten

        Returns:
            Film-Objekt oder None bei Fehler
        """
        try:
            # Savepoint um nur diesen Film zurückzurollen, nicht die gesamte Session
            with self.session.begin_nested():
                # Genres lesen ohne Dict zu mutieren
                genre_names = film_data.get("genres", [])

                # Film erstellen
                film = Film(
                    imdb_id=film_data.get("imdb_id"),
                    titel=film_data.get("titel"),
                    original_titel=film_data.get("original_titel"),
                    jahr=film_data.get("jahr"),
                    bewertung=film_data.get("bewertung"),
                    anzahl_bewertungen=film_data.get("anzahl_bewertungen"),
                    dauer_minuten=film_data.get("dauer_minuten"),
                    beschreibung=film_data.get("beschreibung"),
                    regisseur=film_data.get("regisseur"),
                    poster_url=film_data.get("poster_url"),
                    imdb_url=film_data.get("imdb_url"),
                    rang=film_data.get("rang"),
                    raw_json=film_data.get("raw_json"),
                )

                # Genres hinzufügen
                for genre_name in genre_names:
                    if genre_name:
                        genre = self.get_or_create_genre(genre_name.strip())
                        film.genres.append(genre)

                self.session.add(film)
                self.session.flush()

                # Besetzung (m:n) hinzufügen, nachdem der Film eine ID hat
                self._set_besetzung(film, film_data.get("besetzung", []))
                self.session.flush()

            logger.info(f"Film erstellt: {film.titel} ({film.imdb_id})")
            return film

        except IntegrityError as e:
            logger.warning(f"Film existiert bereits: {film_data.get('imdb_id')} - {e}")
            return None
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Films: {e}")
            raise

    def update_film(self, imdb_id: str, film_data: Dict[str, Any]) -> Optional[Film]:
        """
        Aktualisiert einen bestehenden Film.

        Args:
            imdb_id: IMDb ID des Films
            film_data: Neue Daten

        Returns:
            Aktualisierter Film oder None
        """
        film = self.get_by_imdb_id(imdb_id)
        if not film:
            return None

        # Genres und Besetzung lesen ohne Dict zu mutieren
        genre_names = film_data.get("genres", None)
        besetzung = film_data.get("besetzung", None)

        # Felder aktualisieren (genres/besetzung separat als Relationship)
        for key, value in film_data.items():
            if key in ("genres", "besetzung"):
                continue
            if hasattr(film, key) and value is not None:
                setattr(film, key, value)

        # Genres aktualisieren
        if genre_names is not None:
            film.genres.clear()
            for genre_name in genre_names:
                if genre_name:
                    genre = self.get_or_create_genre(genre_name.strip())
                    film.genres.append(genre)

        # Besetzung (m:n) aktualisieren
        if besetzung is not None:
            self._set_besetzung(film, besetzung)

        self.session.flush()
        logger.info(f"Film aktualisiert: {film.titel}")
        return film

    def upsert_film(self, film_data: Dict[str, Any]) -> Optional[Film]:
        """
        Erstellt oder aktualisiert einen Film.

        Args:
            film_data: Film-Daten

        Returns:
            Film-Objekt
        """
        imdb_id = film_data.get("imdb_id")
        existing = self.get_by_imdb_id(imdb_id)

        if existing:
            return self.update_film(imdb_id, film_data)
        else:
            return self.create_film(film_data)

    def get_by_id(self, film_id: int) -> Optional[Film]:
        """Holt Film nach ID."""
        return self.session.query(Film).filter(Film.id == film_id).first()

    def get_by_imdb_id(self, imdb_id: str) -> Optional[Film]:
        """Holt Film nach IMDb ID."""
        return self.session.query(Film).filter(Film.imdb_id == imdb_id).first()

    def get_all(self, limit: int = None, offset: int = 0) -> List[Film]:
        """Holt alle Filme."""
        query = self.session.query(Film).order_by(Film.rang)
        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)
        return query.all()

    def get_top_rated(self, limit: int = 10) -> List[Film]:
        """Holt die bestbewerteten Filme."""
        return (
            self.session.query(Film)
            .order_by(Film.bewertung.desc())
            .limit(limit)
            .all()
        )

    def get_by_genre(self, genre_name: str) -> List[Film]:
        """Holt Filme nach Genre."""
        return (
            self.session.query(Film)
            .join(Film.genres)
            .filter(Genre.name == genre_name)
            .all()
        )

    def get_by_year(self, year: int) -> List[Film]:
        """Holt Filme nach Jahr."""
        return self.session.query(Film).filter(Film.jahr == year).all()

    def count(self) -> int:
        """Zählt alle Filme."""
        return self.session.query(func.count(Film.id)).scalar()

    def delete(self, imdb_id: str) -> bool:
        """Löscht einen Film."""
        film = self.get_by_imdb_id(imdb_id)
        if film:
            self.session.delete(film)
            self.session.flush()
            logger.info(f"Film gelöscht: {imdb_id}")
            return True
        return False

    def exists(self, imdb_id: str) -> bool:
        """Prüft ob Film existiert."""
        return self.get_by_imdb_id(imdb_id) is not None

    def get_statistics(self) -> Dict[str, Any]:
        """Holt Statistiken über die Filme."""
        return {
            "total": self.count(),
            "avg_rating": self.session.query(func.avg(Film.bewertung)).scalar(),
            "min_year": self.session.query(func.min(Film.jahr)).scalar(),
            "max_year": self.session.query(func.max(Film.jahr)).scalar(),
        }
