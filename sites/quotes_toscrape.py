"""
Site-Konfiguration für quotes.toscrape.com
Entspricht Güngörs Testdomäne 1 (list-Seite, 3 Felder).
"""
from sites.site_config import SiteConfig, FieldConfig

QUOTES_CONFIG = SiteConfig(
    name="quotes.toscrape.com",
    list_url="http://quotes.toscrape.com/page/1/",
    item_selector="div.quote",
    pagination_template="http://quotes.toscrape.com/page/{page}/",
    max_pages=10,
    delay=0.5,
    ground_truth_path="evaluation/ground_truth_quotes.json",
    fields=[
        FieldConfig(
            name="zitat",
            description="Das vollständige Zitat (String, inklusive Anführungszeichen)",
            type="string",
            example="\u201cThe world as we have created it is a process of our thinking.\u201d",
        ),
        FieldConfig(
            name="autor",
            description="Name des Autors/der Autorin (String)",
            type="string",
            example="Albert Einstein",
        ),
        FieldConfig(
            name="tags",
            description="Liste der Tags/Schlagwörter (Array von Strings)",
            type="array",
            example=["change", "deep-thoughts", "thinking", "world"],
        ),
    ],
)
