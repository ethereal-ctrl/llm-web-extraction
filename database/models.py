"""
SQLAlchemy Models für IMDb Filme
"""
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    ForeignKey,
    Table,
    JSON,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column


class Base(DeclarativeBase):
    """Basis-Klasse für alle Models"""
    pass


# Many-to-Many Beziehung: Film <-> Genre
film_genres = Table(
    "film_genres",
    Base.metadata,
    Column("film_id", Integer, ForeignKey("filme.id", ondelete="CASCADE"), primary_key=True),
    Column("genre_id", Integer, ForeignKey("genres.id", ondelete="CASCADE"), primary_key=True),
)


class Genre(Base):
    """Genre Model"""
    __tablename__ = "genres"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)

    # Beziehung zu Filmen
    filme: Mapped[List["Film"]] = relationship(
        "Film",
        secondary=film_genres,
        back_populates="genres",
    )

    def __repr__(self) -> str:
        return f"<Genre(id={self.id}, name='{self.name}')>"


class Person(Base):
    """Person Model (Schauspieler).

    Bildet zusammen mit FilmPerson die m:n-Beziehung Film <-> Person ab
    (ein Film hat viele Schauspieler, ein Schauspieler spielt in vielen
    Filmen). Das Pendant zur Genre-Beziehung, hier aber mit einem
    Zusatzattribut auf der Verbindungstabelle (als_charakter).
    """
    __tablename__ = "personen"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(300), unique=True, nullable=False, index=True)

    # Beziehung zu Filmen (über das Association Object FilmPerson)
    film_rollen: Mapped[List["FilmPerson"]] = relationship(
        "FilmPerson",
        back_populates="person",
    )

    def __repr__(self) -> str:
        return f"<Person(id={self.id}, name='{self.name}')>"


class FilmPerson(Base):
    """Association Object für die m:n-Beziehung Film <-> Person.

    Entspricht der actors2movies-Tabelle: Es verbindet einen Film mit
    einem Schauspieler und hält zusätzlich die gespielte Rolle
    (als_charakter) sowie die Reihenfolge im Cast (ordnung).
    Ein reines secondary=Table genügt hier nicht, weil die Verbindung
    eigene Attribute trägt.
    """
    __tablename__ = "film_personen"

    film_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("filme.id", ondelete="CASCADE"), primary_key=True
    )
    person_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("personen.id", ondelete="CASCADE"), primary_key=True
    )
    # Zusatzattribut der Beziehung: die im Film gespielte Rolle
    als_charakter: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    # Position in der Besetzungsliste (0 = Hauptrolle)
    ordnung: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    film: Mapped["Film"] = relationship("Film", back_populates="besetzung")
    person: Mapped["Person"] = relationship("Person", back_populates="film_rollen")

    def __repr__(self) -> str:
        return (
            f"<FilmPerson(film_id={self.film_id}, person_id={self.person_id}, "
            f"als_charakter='{self.als_charakter}')>"
        )


class Film(Base):
    """Film Model - Haupttabelle für IMDb Filme"""
    __tablename__ = "filme"

    # Primärschlüssel
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # IMDb Identifikation
    imdb_id: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)

    # Film-Informationen
    titel: Mapped[str] = mapped_column(String(500), nullable=False)
    original_titel: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    jahr: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)

    # Bewertungen
    bewertung: Mapped[Optional[float]] = mapped_column(Float, nullable=True, index=True)
    anzahl_bewertungen: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Details
    dauer_minuten: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    beschreibung: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Crew
    regisseur: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Medien
    poster_url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    imdb_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Rang in Top 250
    rang: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)

    # Rohdaten vom LLM (für Debugging)
    raw_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Metadaten
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Beziehung zu Genres
    genres: Mapped[List[Genre]] = relationship(
        "Genre",
        secondary=film_genres,
        back_populates="filme",
    )

    # Beziehung zur Besetzung (m:n über das Association Object FilmPerson)
    besetzung: Mapped[List["FilmPerson"]] = relationship(
        "FilmPerson",
        back_populates="film",
        cascade="all, delete-orphan",
        order_by="FilmPerson.ordnung",
    )

    # Indizes für häufige Abfragen
    __table_args__ = (
        Index("ix_filme_bewertung_jahr", "bewertung", "jahr"),
    )

    def __repr__(self) -> str:
        return f"<Film(id={self.id}, titel='{self.titel}', jahr={self.jahr}, bewertung={self.bewertung})>"

    def to_dict(self) -> dict:
        """Konvertiert Film zu Dictionary"""
        return {
            "id": self.id,
            "imdb_id": self.imdb_id,
            "titel": self.titel,
            "original_titel": self.original_titel,
            "jahr": self.jahr,
            "bewertung": self.bewertung,
            "anzahl_bewertungen": self.anzahl_bewertungen,
            "dauer_minuten": self.dauer_minuten,
            "beschreibung": self.beschreibung,
            "regisseur": self.regisseur,
            "genres": [g.name for g in self.genres],
            "besetzung": [
                {"name": fp.person.name, "als_charakter": fp.als_charakter}
                for fp in self.besetzung
            ],
            "poster_url": self.poster_url,
            "imdb_url": self.imdb_url,
            "rang": self.rang,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
