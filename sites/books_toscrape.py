"""
Site-Konfiguration für books.toscrape.com (Listenseite)
Entspricht Güngörs Testdomäne 2 (list-Seite, 4 Felder).
"""
from sites.site_config import SiteConfig, FieldConfig

BOOKS_LIST_CONFIG = SiteConfig(
    name="books.toscrape.com (list)",
    list_url="http://books.toscrape.com/catalogue/page-1.html",
    item_selector="article.product_pod",
    pagination_template="http://books.toscrape.com/catalogue/page-{page}.html",
    max_pages=3,   # 3 Seiten × 20 Bücher = 60 Items (Güngör: 50 Seiten)
    delay=0.5,
    ground_truth_path="evaluation/ground_truth_books_list.json",
    fields=[
        FieldConfig(
            name="titel",
            description="Buchtitel (String)",
            type="string",
            example="A Light in the Attic",
        ),
        FieldConfig(
            name="bewertung",
            description="Sternebewertung als Zahl 1-5 (Integer). Lies den CSS-Klassennamen des Elements <p class='star-rating X'> – der zweite Klassenname gibt die Bewertung an: One=1, Two=2, Three=3, Four=4, Five=5. Ignoriere die Anzahl der <i>-Icons (immer 5).",
            type="int",
            example=3,
        ),
        FieldConfig(
            name="preis",
            description="Preis als Dezimalzahl in GBP ohne Währungssymbol (Float)",
            type="float",
            example=51.77,
        ),
        FieldConfig(
            name="verfuegbarkeit",
            description="Verfügbarkeit: true wenn 'In stock', false wenn 'Out of stock' (Boolean)",
            type="bool",
            example=True,
        ),
    ],
)
