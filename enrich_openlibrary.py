"""
Step 2 — Enrich with Open Library (ONLY title, author, genre).
Input : data/books_raw.csv (columns: title, category)
Output: data/books.csv      (columns: title, author, genre)
"""

import os
import time
import csv
import requests
import pandas as pd
from typing import Optional, Tuple

IN_PATH  = "data/books_raw.csv"
OUT_PATH = "data/books.csv"
CACHE_PATH = "data/ol_cache.csv"   # optional cache to avoid re-querying same titles

HDRS = {"User-Agent": "BookRecsBootcamp/1.0"}
URL  = "https://openlibrary.org/search.json"


def fetch_author_genre(title: str) -> Tuple[Optional[str], Optional[str]]:
    """Query Open Library by title, return (author, genre)."""
    try:
        r = requests.get(URL, params={"title": title, "limit": 3},
                         headers=HDRS, timeout=25)
        r.raise_for_status()
        js = r.json()
        if not js.get("docs"):
            return None, None
        # choose the doc with highest edition_count as a simple quality proxy
        best = max(js["docs"], key=lambda d: d.get("edition_count", 0))
        author = (best.get("author_name") or [None])[0]
        subjects = best.get("subject") or []
        # pick a short subject as genre (fallbacks handled later)
        genre = next((s for s in subjects if isinstance(s, str) and 2 <= len(s) <= 40), None)
        return author, genre
    except Exception:
        return None, None


def load_cache() -> dict:
    if not os.path.exists(CACHE_PATH):
        return {}
    cache = {}
    with open(CACHE_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cache[row["title"]] = (row.get("author") or None, row.get("genre") or None)
    return cache


def save_cache(cache: dict) -> None:
    os.makedirs("data", exist_ok=True)
    with open(CACHE_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["title", "author", "genre"])
        for t, (a, g) in cache.items():
            w.writerow([t, a or "", g or ""])


if __name__ == "__main__":
    if (not os.path.exists(IN_PATH)) or os.path.getsize(IN_PATH) == 0:
        raise FileNotFoundError("Missing/empty data/books_raw.csv. Run scraping first.")

    raw = pd.read_csv(IN_PATH)
    # Drop obvious dup titles early (keeps first category only)
    titles = raw["title"].astype(str).drop_duplicates().tolist()

    cache = load_cache()
    out_rows = []

    for i, title in enumerate(titles, start=1):
        if title in cache:
            author, genre = cache[title]
        else:
            author, genre = fetch_author_genre(title)
            cache[title] = (author, genre)
            time.sleep(0.2)  # polite rate limit

        # Fallbacks
        if not genre:
            # if raw had a category for this title, use the first one found
            cat_match = raw.loc[raw["title"] == title, "category"]
            genre = cat_match.iloc[0] if len(cat_match) else "Unknown"
        if not author:
            author = "Unknown"

        out_rows.append({"title": title, "author": author, "genre": genre})

        if i % 25 == 0:
            print(f"Enriched {i}/{len(titles)}")
            save_cache(cache)  # periodic cache save

    # Final save
    df = pd.DataFrame(out_rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    save_cache(cache)
    print(f"Saved {len(df)} rows -> {OUT_PATH}")
