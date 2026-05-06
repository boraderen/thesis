import csv
import genanki
import os
from datetime import date

FLASHCARDS_CSV = os.path.join(os.path.dirname(__file__), "flashcards.csv")
APKG_OUT       = os.path.join(os.path.dirname(__file__), "thesis_study.apkg")
MEDIA_DIR      = os.path.join(os.path.dirname(__file__), "media")

MODEL_ID = 1_607_392_323
DECK_ID  = 2_059_400_114

model = genanki.Model(
    MODEL_ID,
    "Thesis Study Model",
    fields=[
        {"name": "Front"},
        {"name": "Back"},
        {"name": "Category"},
        {"name": "Source"},
        {"name": "Notes"},
    ],
    templates=[
        {
            "name": "Card",
            "qfmt": """
<div class="category">{{Category}}</div>
<h2>{{Front}}</h2>
""",
            "afmt": """
<div class="category">{{Category}}</div>
<h2>{{Front}}</h2>
<hr>
{{Back}}
{{#Source}}<div class="source">📄 {{Source}}</div>{{/Source}}
{{#Notes}}<div class="notes">💡 {{Notes}}</div>{{/Notes}}
""",
        }
    ],
    css="""
.card { font-family: Georgia, serif; font-size: 16px; max-width: 650px; margin: auto; padding: 16px; }
h2 { color: #2c3e50; margin-bottom: 8px; }
.category { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #888; margin-bottom: 6px; }
.source { margin-top: 16px; font-size: 13px; color: #666; font-style: italic; }
.notes { margin-top: 8px; font-size: 13px; color: #555; border-left: 3px solid #ccc; padding-left: 8px; }
hr { border: none; border-top: 1px solid #eee; margin: 12px 0; }
""",
)

deck = genanki.Deck(DECK_ID, "Thesis Study")

with open(FLASHCARDS_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    count = 0
    for row in reader:
        note = genanki.Note(
            model=model,
            fields=[
                row.get("Front", ""),
                row.get("Back", ""),
                row.get("Category", ""),
                row.get("Source", ""),
                row.get("Notes", ""),
            ],
        )
        deck.add_note(note)
        count += 1

genanki.Package(deck).write_to_file(APKG_OUT)
print(f"Wrote {APKG_OUT} ({count} card(s))")
