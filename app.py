# app.py
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Book Recommendation System")
st.caption("Find books similar to your favorite title (based on title, author, genre).")

# ----------------------------
# Data & model (cached)
# ----------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/books_clean.csv")
    # normalize lowercase title for fast lookup (kept only in memory)
    df["title_norm"] = df["title"].str.lower().str.strip()
    return df

@st.cache_resource
def build_vectorizer(df: pd.DataFrame):
    corpus = (df["title"].fillna("") + " " +
              df["author"].fillna("") + " " +
              df["genre"].fillna(""))
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = tfidf.fit_transform(corpus)
    title_to_idx = {t: i for i, t in enumerate(df["title_norm"])}
    return tfidf, X, title_to_idx

df = load_data()
tfidf, X, title_to_idx = build_vectorizer(df)

# ----------------------------
# Recommender
# ----------------------------
def recommend_by_title(
    title: str,
    top_k: int = 10,
    genre_filter: str | None = None
) -> pd.DataFrame:
    """Return up to top_k most similar books to the given title."""
    q = (title or "").lower().strip()
    if q not in title_to_idx:
        # fallback: contains search
        cand = df[df["title_norm"].str.contains(q, na=False)]
        if cand.empty:
            return df.iloc[0:0][["title", "author", "genre", "similarity"]]
        q = cand.iloc[0]["title_norm"]

    idx = title_to_idx[q]
    sims = cosine_similarity(X[idx], X).ravel()

    # exclude the query itself
    sims[idx] = -1.0

    # take a larger pool first, then post-filter & cut to top_k
    pool_idx = np.argsort(-sims)[: top_k * 3]
    res = df.iloc[pool_idx][["title", "author", "genre"]].copy()
    res["similarity"] = np.round(sims[pool_idx], 3)

    # keep only positive similarities
    res = res[res["similarity"] > 0]

    # optional genre filter
    if genre_filter and genre_filter != "All":
        res = res[res["genre"] == genre_filter]

    # final top_k
    res = res.sort_values("similarity", ascending=False).head(top_k)
    return res.reset_index(drop=True)

# ----------------------------
# UI controls
# ----------------------------
left, right = st.columns([2, 1])

with left:
    title_input = st.selectbox(
        "Choose a book title",
        options=sorted(df["title"].unique().tolist()),
        index=None,
        placeholder="Start typing a title..."
    )

with right:
    top_k = st.slider("Number of recommendations", min_value=3, max_value=20, value=10, step=1)

genre_choice = st.selectbox(
    "Optional: filter by genre",
    options=["All"] + sorted(df["genre"].dropna().unique().tolist())
)

go = st.button("✅ Recommend books", use_container_width=False)

# ----------------------------
# Results
# ----------------------------
if go:
    if not title_input:
        st.warning("Please choose a title first.")
    else:
        recs = recommend_by_title(title_input, top_k=top_k, genre_filter=genre_choice)
        if recs.empty:
            st.warning("No matches found. Try another title or remove the genre filter.")
        else:
            # Pretty table (rounded similarity)
            st.dataframe(
                recs.style.format({"similarity": "{:.3f}"}),
                use_container_width=True,
                hide_index=True
            )
            # Download button
            csv = recs.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download recommendations (CSV)",
                data=csv,
                file_name="recommendations.csv",
                mime="text/csv"
            )

# ----------------------------
# Dataset stats
# ----------------------------
with st.expander("📊 Dataset stats", expanded=False):
    st.write(
        f"Rows: **{len(df)}** | Unique authors: **{df['author'].nunique()}** | "
        f"Unique genres: **{df['genre'].nunique()}**"
    )
    st.write(f"Unknown authors %: **{(df['author'] == 'Unknown').mean():.2%}**")

    # Top-10 genres (bar chart)
    top_genres = df["genre"].value_counts().head(10).sort_values(ascending=True)
    st.bar_chart(top_genres)