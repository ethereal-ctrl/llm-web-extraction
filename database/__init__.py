"""
Database Modul - PostgreSQL Anbindung für IMDb Filme
"""
from .models import Base, Film, Genre, film_genres
from .connection import DatabaseConnection, get_engine, get_session
from .repository import FilmRepository

__all__ = [
    "Base",
    "Film",
    "Genre",
    "film_genres",
    "DatabaseConnection",
    "get_engine",
    "get_session",
    "FilmRepository",
]
