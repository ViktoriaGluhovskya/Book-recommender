"""
Enrich books with AUTHOR (and better GENRE) using:
1) Open Library Search API (robust matching)
2) Fallback: Google Books API
Input : data/books_raw.csv  (title, category)
Output: data/books.csv      (title, author, genre)
"""

import os, time, re, csv
import requests, pandas as pd
from difflib import SequenceMatcher
from typing import Optional, Tuple, Dict, Any

IN_PATH   = "data/books_raw.csv"
OUT_PATH  = "data/books.csv"
CACHE_PATH = "data/enrich_cache.csv"

OL_URL = "https://openlibrary.org/search.json"
GB_URL = "https://www.googleapis.com/books/v1/volumes"
HDRS   = {"User-Agent": "BookRecsBootcamp/1.0"}

def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()

def ol_fetch(title: str) -> Tuple[Optional[str], Optional[str], float]:
    """Return (author, genre, score) from Open Library best match."""
    try:
        r = requests.get(OL_URL,
                         params={"title": title, "limit": 5, "language": "eng"},
                         headers=HDRS, timeout=25)
        r.raise_for_status()
        docs = r.json().get("docs") or []
        best = (None, None, 0.0)
        for d in docs:
            cand_title = d.get("title") or ""
            sim = title_similarity(title, cand_title)
            ed  = d.get("edition_count", 0) or 0
            score = sim * 0.7 + (min(ed, 50) / 50.0) * 0.3  # combine similarity + popularity
            if score > best[2]:
                author = (d.get("author_name") or [None])[0]
                subs   = d.get("subject") or []
                genre  = next((s for s in subs if isinstance(s,str) and 2 <= len(s) <= 40), None)
                best = (author, genre, score)
        return best
    except Exception:
        return None, None, 0.0

def gb_fetch(title: str) -> Tuple[Optional[str], Optional[str]]:
    """Fallback: Google Books API (no key needed for basic search)."""
    try:
        r = requests.get(GB_URL, params={"q": f"intitle:{title}", "maxResults": 5}, timeout=25)
        r.raise_for_status()
        items = (r.json().get("items") or [])
        bestA, bestG, bestScore = None, None, 0.0
        for it in items:
            info = it.get("volumeInfo", {})
            cand_title = info.get("title") or ""
            sim = title_similarity(title, cand_title)
            rating = info.get("averageRating", 0) or 0
            score = sim * 0.8 + (rating/5.0)*0.2
            if score > bestScore:
                authors = info.get("authors") or []
                cats    = info.get("categories") or []
                bestA   = authors[0] if authors else None
                bestG   = cats[0] if cats else None
                bestScore = score
        return bestA, bestG
    except Exception:
        return None, None

def load_cache() -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    if not os.path.exists(CACHE_PATH): return {}
    cache = {}
    with open(CACHE_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cache[row["title"]] = (row.get("author") or None, row.get("genre") or None)
    return cache

def save_cache(cache: Dict[str, Tuple[Optional[str], Optional[str]]]) -> None:
    os.makedirs("data", exist_ok=True)
    with open(CACHE_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["title","author","genre"])
        for t,(a,g) in cache.items():
            w.writerow([t, a or "", g or ""])

if __name__ == "__main__":
    if (not os.path.exists(IN_PATH)) or os.path.getsize(IN_PATH) == 0:
        raise FileNotFoundError("Missing/empty data/books_raw.csv. Run scraping first.")

    raw = pd.read_csv(IN_PATH)
    titles = raw["title"].astype(str).drop_duplicates().tolist()
    cache = load_cache()
    out = []

    for i, t in enumerate(titles, start=1):
        if t in cache:
            a, g = cache[t]
        else:
            # try Open Library
            a, g, score = ol_fetch(t)
            # fallback to Google Books if author missing
            if not a:
                a2, g2 = gb_fetch(t)
                a = a2 or a
                g = g or g2
            # final fallbacks
            if not g:
                cat_series = raw.loc[raw["title"] == t, "category"]
                g = cat_series.iloc[0] if len(cat_series) else "Unknown"
            if not a:
                a = "Unknown"
            cache[t] = (a, g)
            time.sleep(0.2)

        out.append({"title": t, "author": a, "genre": g})

        if i % 25 == 0:
            print(f"Enriched {i}/{len(titles)}")
            save_cache(cache)

    df = pd.DataFrame(out)
    df.to_csv(OUT_PATH, index=False)
    save_cache(cache)
    print(f"Saved {len(df)} rows -> {OUT_PATH}")
