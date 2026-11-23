"""
Site-Konfiguration für letterboxd.com (Filmdetailseiten).

Letterboxd ist eine kommerzielle, dynamisch gerenderte Filmplattform mit
einem strukturell anderen DOM als IMDb. JSON-LD-Strukturdaten sind
vorhanden, sollen aber NICHT vom Cleaner exklusiv ausgewertet werden —
das wäre Wrapper-Induktion. Stattdessen filtert der generische Cleaner
unstrukturierte Tags und übergibt den verbleibenden Klartext dem LLM.

Anders als IMDb liefert Letterboxd schon ohne JavaScript-Rendering die
relevanten Felder im HTML — der Scraper kann daher auf requests/BS4
laufen statt Playwright zu starten.
"""
from sites.site_config import SiteConfig, FieldConfig

LETTERBOXD_CONFIG = SiteConfig(
    name="letterboxd.com",
    list_url="https://letterboxd.com/films/by/popular/",
    item_selector="li.poster-container",
    pagination_template=None,
    max_pages=1,
    delay=2.0,
    ground_truth_path="evaluation/ground_truth_letterboxd.json",
    fields=[
        FieldConfig(
            name="titel",
            description="Filmtitel (String)",
            type="string",
            example="The Shawshank Redemption",
        ),
        FieldConfig(
            name="jahr",
            description="Erscheinungsjahr (Integer, vierstellig)",
            type="int",
            example=1994,
        ),
        FieldConfig(
            name="regisseur",
            description="Name des Regisseurs (String)",
            type="string",
            example="Frank Darabont",
        ),
        FieldConfig(
            name="bewertung",
            description=(
                "Letterboxd Durchschnittsbewertung (Float, 1.0–5.0). "
                "ACHTUNG: Letterboxd-Skala 0–5, NICHT IMDb-Skala 0–10."
            ),
            type="float",
            example=4.59,
        ),
        FieldConfig(
            name="dauer_minuten",
            description=(
                "Filmlänge in Minuten als Zahl (Integer, z. B. 142). "
                "Auf Letterboxd typisch im Format 'XXX mins'."
            ),
            type="int",
            example=142,
        ),
        FieldConfig(
            name="genres",
            description=(
                "Liste der Genres (Array von Strings). Verwende NUR die "
                "offiziellen Letterboxd-Genres aus dieser Liste: Action, "
                "Adventure, Animation, Comedy, Crime, Documentary, Drama, "
                "Family, Fantasy, History, Horror, Music, Mystery, Romance, "
                "Science Fiction, Thriller, TV Movie, War, Western. "
                "Ignoriere Themes/Schlagwörter."
            ),
            type="array",
            example=["Crime", "Drama"],
        ),
    ],
)
