# import_papers.py
import csv
from app import start_app, db
from models import Paper  # adjust if your model lives elsewhere

app = start_app()

with app.app_context():
    with open("data/papers_trimmed.csv", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for id_num, row in enumerate(reader):
            paper = Paper(
                id = id_num,
                arxiv_id=row["id"],
                title=row["title"],
                authors=row["authors"],
                url=row["url"]
            )
            db.session.add(paper)
        db.session.commit()
    print("✅ Papers imported!")