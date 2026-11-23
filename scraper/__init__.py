"""
Scraper Modul - Web Scraping für IMDb
"""
from .spider import IMDbScraper
from .cleaner import HTMLCleaner

__all__ = ["IMDbScraper", "HTMLCleaner"]
