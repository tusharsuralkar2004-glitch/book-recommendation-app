import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Page setup ----------
st.set_page_config(page_title="Book Recommendation System", page_icon="📚", layout="centered")

# ---------- Custom CSS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Poppins:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* App background */
    .stApp {
        background: linear-gradient(180deg, #0f1c2e 0%, #16283f 45%, #1c3452 100%);
        color: #eef2f7;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* Hero header */
    .hero-wrap {
        text-align: center;
        padding: 2.2rem 1rem 1.6rem 1rem;
        margin-bottom: 1.2rem;
        border-radius: 18px;
        background: radial-gradient(circle at top, rgba(28,114,147,0.35), transparent 70%);
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: #f4d58d;
        margin-bottom: 0.3rem;
        letter-spacing: 0.5px;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        font-weight: 300;
        color: #cdd9e5;
        max-width: 520px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Text area */
    .stTextArea textarea {
        background-color: #10233a !important;
        color: #f1f3f6 !important;
        border: 1.5px solid #2f5372 !important;
        border-radius: 12px !important;
        font-size: 0.98rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #f4d58d !important;
        box-shadow: 0 0 0 1px #f4d58d !important;
    }
    .stTextArea label {
        color: #cdd9e5 !important;
        font-weight: 500 !important;
    }

    /* Primary button */
    .stButton > button {
        background: linear-gradient(135deg, #1c7293, #0f4c5c) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 0.6rem 1.8rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.4px;
        box-shadow: 0 4px 14px rgba(28, 114, 147, 0.45);
        transition: all 0.25s ease-in-out;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(244, 213, 141, 0.35);
        color: #0f1c2e !important;
        background: linear-gradient(135deg, #f4d58d, #e8b85f) !important;
    }

    /* Success / genre badge */
    .stAlert {
        background-color: rgba(244, 213, 141, 0.12) !important;
        border: 1px solid #f4d58d !important;
        border-radius: 12px !important;
        color: #f4d58d !important;
    }

    .section-heading {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        color: #f4d58d;
        border-bottom: 1px solid rgba(244, 213, 141, 0.35);
        padding-bottom: 0.4rem;
        margin-top: 1.4rem;
        margin-bottom: 1rem;
    }

    /* Book card */
    .book-card {
        display: flex;
        gap: 1rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease-in-out;
    }
    .book-card:hover {
        border-color: rgba(244, 213, 141, 0.5);
        background: rgba(255, 255, 255, 0.06);
        transform: translateY(-1px);
    }
    .book-card img {
        border-radius: 8px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.4);
        object-fit: cover;
    }
    .book-rank {
        display: inline-block;
        background: #f4d58d;
        color: #0f1c2e;
        font-weight: 700;
        font-size: 0.78rem;
        border-radius: 50%;
        width: 22px;
        height: 22px;
        line-height: 22px;
        text-align: center;
        margin-right: 6px;
    }
    .book-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.15rem;
        color: #f8f7f2;
        margin-bottom: 0.15rem;
    }
    .book-author {
        font-size: 0.85rem;
        color: #9fb3c8;
        font-style: italic;
        margin-bottom: 0.5rem;
    }
    .book-desc {
        font-size: 0.85rem;
        color: #cdd9e5;
        line-height: 1.5;
    }

    .stLinkButton > a {
        background: transparent !important;
        border: 1.5px solid #f4d58d !important;
        color: #f4d58d !important;
        border-radius: 20px !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
    }
    .stLinkButton > a:hover {
        background: #f4d58d !important;
        color: #0f1c2e !important;
    }

    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
    }

    hr {
        border-color: rgba(255,255,255,0.08) !important;
    }

    /* ---------- Search-engine decorations ---------- */

    /* Faint floating book/page pattern behind everything */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        opacity: 0.05;
        background-image:
            radial-gradient(circle at 8% 15%, #f4d58d 0px, #f4d58d 2px, transparent 3px),
            radial-gradient(circle at 85% 10%, #f4d58d 0px, #f4d58d 2px, transparent 3px),
            radial-gradient(circle at 92% 80%, #f4d58d 0px, #f4d58d 2px, transparent 3px),
            radial-gradient(circle at 15% 85%, #f4d58d 0px, #f4d58d 2px, transparent 3px);
        background-size: 100% 100%;
    }

    /* Top navbar, like a bookstore search site */
    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.9rem 1.4rem;
        margin: -1rem -1rem 1.6rem -1rem;
        background: rgba(10, 20, 34, 0.65);
        backdrop-filter: blur(6px);
        border-bottom: 1px solid rgba(244, 213, 141, 0.18);
        position: sticky;
        top: 0;
        z-index: 999;
    }
    .navbar-logo {
        font-family: 'Playfair Display', serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: #f4d58d;
        letter-spacing: 0.3px;
    }
    .navbar-tag {
        font-size: 0.75rem;
        color: #9fb3c8;
        font-style: italic;
    }

    /* Make the description box look like a search bar */
    div[data-testid="stTextArea"] {
        position: relative;
    }
    div[data-testid="stTextArea"]::before {
        content: "🔍";
        position: absolute;
        top: 14px;
        left: 14px;
        font-size: 1.1rem;
        z-index: 2;
        opacity: 0.85;
    }
    div[data-testid="stTextArea"] textarea {
        padding-left: 2.4rem !important;
    }

    /* "Searching" style results meta line */
    .results-meta {
        text-align: center;
        color: #9fb3c8;
        font-size: 0.85rem;
        margin-bottom: 0.6rem;
        letter-spacing: 0.2px;
    }
    .results-meta b {
        color: #f4d58d;
    }

    /* Card entrance animation, staggered like search results loading in */
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .book-card {
        animation: fadeSlideIn 0.45s ease-out both;
    }

    /* Little "match" ribbon on each card */
    .match-chip {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 600;
        color: #0f1c2e;
        background: #f4d58d;
        padding: 0.12rem 0.55rem;
        border-radius: 10px;
        margin-left: 0.4rem;
        vertical-align: middle;
        letter-spacing: 0.3px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Navbar ----------
st.markdown("""
<div class="navbar">
    <div class="navbar-logo">📖 BookFinder</div>
    <div class="navbar-tag">discover your next read</div>
</div>
""", unsafe_allow_html=True)

# ---------- Load saved model, vectorizer, and data ----------
@st.cache_resource
def load_artifacts():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    return model, vectorizer, data

model, vectorizer, data = load_artifacts()

# Pre-compute TF-IDF for the whole catalog once (faster at runtime)
@st.cache_resource
def precompute_tfidf(_vectorizer, _data):
    return _vectorizer.transform(_data['Description'])

catalog_tfidf = precompute_tfidf(vectorizer, data)

# ---------- Recommendation logic ----------
def recommend_books(description, top_n=5):
    new_tfidf = vectorizer.transform([description])
    predicted_genre = model.predict(new_tfidf)[0]

    genre_mask = data['Main_Genre'] == predicted_genre
    genre_books = data[genre_mask].reset_index(drop=True)
    genre_tfidf = catalog_tfidf[genre_mask.values]

    similarities = cosine_similarity(new_tfidf, genre_tfidf).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]

    results = genre_books.iloc[top_indices][['Title', 'Authors', 'Description']].reset_index(drop=True)
    return predicted_genre, results

@st.cache_data(ttl=3600, show_spinner=False)
def get_book_details(title, author, fallback_description=""):
    import requests
    import urllib.parse
    info_link = None
    description = fallback_description
    image_url = None

    try:
        query = f"intitle:{title} inauthor:{author}"
        url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=1"
        response = requests.get(url, timeout=3)
        data = response.json()
        if data.get("totalItems", 0) > 0:
            book_info = data["items"][0]["volumeInfo"]
            info_link = book_info.get("infoLink") or book_info.get("previewLink")
            api_desc = book_info.get("description", "")
            if api_desc:
                description = api_desc
            image_links = book_info.get("imageLinks", {})
            raw_image = image_links.get("thumbnail") or image_links.get("smallThumbnail")
            if raw_image:
                image_url = raw_image.replace("http://", "https://")
    except Exception:
        pass

    if not info_link or not image_url:
        try:
            query = urllib.parse.quote(f"{title} {author}")
            url = f"https://openlibrary.org/search.json?q={query}&limit=1"
            response = requests.get(url, timeout=3)
            data = response.json()
            if data.get("docs"):
                if not info_link:
                    key = data["docs"][0].get("key")
                    if key:
                        info_link = f"https://openlibrary.org{key}"
                if not image_url:
                    cover_id = data["docs"][0].get("cover_i")
                    if cover_id:
                        image_url = f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg"
        except Exception:
            pass

    if not info_link:
        search_query = urllib.parse.quote(f"{title} {author}")
        info_link = f"https://www.google.com/search?tbm=bks&q={search_query}"

    
    if not image_url:
        initial = title[0].upper() if title else "B"
        image_url = f"https://placehold.co/90x120/1C7293/FFFFFF?text={initial}"

    shop_query = urllib.parse.quote(f"{title} {author}")
    amazon_link = f"https://www.amazon.in/s?k={shop_query}"

    return info_link, description, amazon_link, image_url
    # ---------- UI ----------
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">📚 Book Recommendation System</div>
    <div class="hero-subtitle">Describe a story you're in the mood for, and get five handpicked
    recommendations matched by genre and content.</div>
</div>
""", unsafe_allow_html=True)

user_input = st.text_area(
    "Book description",
    placeholder="e.g. A thrilling mystery about a detective solving a murder case in a small town...",
    height=120,
    label_visibility="collapsed",
)

_, btn_col, _ = st.columns([1, 1, 1])
with btn_col:
    get_recs = st.button("✨ Get Recommendations", type="primary", use_container_width=True)

if get_recs:
    if user_input.strip() == "":
        st.warning("Please enter a description first.")
    else:
        with st.spinner("🔎 Searching the shelves for your next read..."):
            genre, recommendations = recommend_books(user_input, top_n=5)

        st.markdown(f"""
        <div style="text-align:center; margin: 1.2rem 0;">
            <span style="background: rgba(244,213,141,0.14); border: 1px solid #f4d58d;
            color:#f4d58d; padding: 0.45rem 1.2rem; border-radius: 20px; font-weight:600;
            font-size: 0.95rem;">🔖 Predicted Genre: {genre}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-heading">Search Results</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="results-meta">Showing <b>{len(recommendations)}</b> results '
            f'matched to <b>"{genre}"</b></div>',
            unsafe_allow_html=True,
        )

        for i, row in recommendations.iterrows():
            book_url, description, amazon_link, image_url = get_book_details(
                row['Title'], row['Authors'], row['Description']
            )

            short_desc = description if description and len(description) <= 220 else (description[:220] + "…" if description else "")
            delay = i * 0.08

            st.markdown(f'<div class="book-card" style="animation-delay:{delay}s;">', unsafe_allow_html=True)
            img_col, info_col = st.columns([1, 4])

            with img_col:
                st.markdown('<div style="padding-top:0.4rem;">', unsafe_allow_html=True)
                if image_url:
                    st.image(image_url, width=90)
                else:
                    st.markdown("📚", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with info_col:
                st.markdown(f"""
                <div class="book-title"><span class="book-rank">{i+1}</span>
                <a href="{book_url}" target="_blank" style="color:#f8f7f2; text-decoration:none;">{row['Title']}</a>
                <span class="match-chip">{max(95 - i * 6, 70)}% match</span></div>
                <div class="book-author">by {row['Authors']}</div>
                """, unsafe_allow_html=True)

                if short_desc:
                    st.markdown(f'<div class="book-desc">{short_desc}</div>', unsafe_allow_html=True)
                    if description and len(description) > 220:
                        with st.expander("Read full description"):
                            st.write(description)

                st.link_button("🛒 Buy on Amazon", amazon_link, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)
