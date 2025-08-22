import os, time
from urllib.parse import urljoin
import requests, pandas as pd
from bs4 import BeautifulSoup

BASE = "http://books.toscrape.com/"
CATALOG = urljoin(BASE, "catalogue/")
HDRS = {"User-Agent": "Mozilla/5.0"}

def get_soup(url):
    r = requests.get(url, headers=HDRS, timeout=25)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def get_categories():
    s = get_soup(BASE)
    links = s.select("div.side_categories ul > li > ul > li > a")
    cats = []
    for a in links:
        name = a.get_text(strip=True)
        href = urljoin(BASE, a.get("href"))
        cats.append((name, href))
    return cats

def parse_card(card):
    a = card.select_one("h3 a")
    title = a["title"].strip()
    rel = a["href"]
    href = urljoin(CATALOG, rel)
    return {"title": title, "category": name}

def scrape_category(url, name):
    rows, page, total = [], 1, 0
    while True:
        s = get_soup(url)
        cards = s.select("article.product_pod")
        for c in cards:
            a = c.select_one("h3 a")
            title = a["title"].strip()
            rows.append({"title": title, "category": name})
        total += len(cards)
        nxt = s.select_one("li.next a")
        if not nxt: break
        url = urljoin(url, nxt["href"])
        page += 1
        time.sleep(0.15)
    print(f"  - {name}: {total} books, {page} pages")
    return rows

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    cats = get_categories()
    print("Categories found:", len(cats))

    all_rows = []
    for i, (name, href) in enumerate(cats, start=1):
        print(f"[{i:02d}] {name}")
        all_rows.extend(scrape_category(href, name))

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["title","category"])
    out = "data/books_raw.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows -> {out}")
