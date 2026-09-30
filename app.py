import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Page setup ----------
st.set_page_config(page_title="Inkwell — Book Recommendations", page_icon="📖", layout="centered")

# ---------- Custom CSS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Lora:ital,wght@0,400;0,500;1,400&display=swap');

    html, body, [class*="css"] {
        font-family: 'Lora', serif;
    }

    /* ---------- Warm parchment background with soft light washes ---------- */
    .stApp {
        background:
            radial-gradient(circle at 8% 12%, rgba(201,162,39,0.16), transparent 38%),
            radial-gradient(circle at 92% 88%, rgba(107,31,31,0.14), transparent 42%),
            radial-gradient(circle at 85% 15%, rgba(47,75,60,0.10), transparent 35%),
            linear-gradient(180deg, #FBF3E3 0%, #F3E6C9 100%);
        color: #2B2118;
    }
    #MainMenu, footer, header {visibility: hidden;}

    /* Faint paper-grain dots */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        opacity: 0.55;
        background-image: radial-gradient(rgba(107,31,31,0.05) 1px, transparent 1px);
        background-size: 14px 14px;
    }

    /* ---------- Decorative bookshelf strip behind the navbar ---------- */
    .shelf {
        position: relative;
        height: 34px;
        margin: -1rem -1rem 0 -1rem;
        background-image: repeating-linear-gradient(
            90deg,
            #6B1F1F 0px, #6B1F1F 22px,
            #C9A227 22px, #C9A227 26px,
            #2F4B3C 26px, #2F4B3C 46px,
            #8B3A3A 46px, #8B3A3A 66px,
            #4A6B57 66px, #4A6B57 82px,
            #B08628 82px, #B08628 100px
        );
        border-bottom: 6px solid #3B2A1A;
        box-shadow: 0 3px 8px rgba(0,0,0,0.18);
    }

    /* ---------- Navbar ---------- */
    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.9rem 1.4rem 0.6rem 1.4rem;
        margin: 0 -1rem 1rem -1rem;
        position: relative;
        z-index: 1;
    }
    .navbar-logo {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        font-weight: 800;
        color: #6B1F1F;
        letter-spacing: 0.3px;
    }
    .navbar-tag {
        font-size: 0.78rem;
        color: #7A6A52;
        font-style: italic;
    }

    /* ---------- Hero ---------- */
    .hero-wrap {
        text-align: center;
        padding: 1.6rem 1rem 1.4rem 1rem;
        margin-bottom: 1.4rem;
        position: relative;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: #3B1414;
        margin-bottom: 0.4rem;
        letter-spacing: 0.3px;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        font-weight: 400;
        color: #5C4A34;
        max-width: 480px;
        margin: 0 auto;
        line-height: 1.55;
        font-style: italic;
    }
    .hero-rule {
        width: 90px;
        height: 3px;
        margin: 0.9rem auto 0 auto;
        background: linear-gradient(90deg, transparent, #C9A227, transparent);
    }

    /* ---------- Text area styled like an old library search slip ---------- */
    div[data-testid="stTextArea"] {
        position: relative;
    }
    .stTextArea textarea {
        background-color: #FFFDF7 !important;
        color: #2B2118 !important;
        border: 1.5px solid #C9A227 !important;
        border-radius: 6px !important;
        font-size: 0.98rem !important;
        font-family: 'Lora', serif !important;
        padding-left: 2.5rem !important;
        box-shadow: inset 0 1px 4px rgba(107,31,31,0.08) !important;
    }
    .stTextArea textarea:focus {
        border-color: #6B1F1F !important;
        box-shadow: 0 0 0 1px #6B1F1F !important;
    }
    .stTextArea label { display: none !important; }
    div[data-testid="stTextArea"]::before {
        content: "🔍";
        position: absolute;
        top: 14px;
        left: 14px;
        font-size: 1.05rem;
        z-index: 2;
        opacity: 0.8;
    }

    /* ---------- Primary button — gold foil ---------- */
    .stButton > button {
        background: linear-gradient(135deg, #C9A227, #9C7A1A) !important;
        color: #FBF3E3 !important;
        border: none !important;
        border-radius: 24px !important;
        padding: 0.6rem 1.8rem !important;
        font-weight: 600 !important;
        font-family: 'Lora', serif !important;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 12px rgba(201, 162, 39, 0.4);
        transition: all 0.22s ease-in-out;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(107, 31, 31, 0.3);
        background: linear-gradient(135deg, #6B1F1F, #4A1515) !important;
        color: #F3E6C9 !important;
    }

    /* ---------- Genre badge ---------- */
    .stAlert {
        background-color: rgba(47, 75, 60, 0.08) !important;
        border: 1px solid #2F4B3C !important;
        border-radius: 10px !important;
        color: #2F4B3C !important;
    }

    .section-heading {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #6B1F1F;
        border-bottom: 2px dotted rgba(107, 31, 31, 0.35);
        padding-bottom: 0.4rem;
        margin-top: 1.4rem;
        margin-bottom: 0.9rem;
    }
    .results-meta {
        text-align: center;
        color: #7A6A52;
        font-size: 0.85rem;
        margin-bottom: 0.9rem;
        font-style: italic;
    }
    .results-meta b { color: #6B1F1F; font-style: normal; }

    /* ---------- Book card — looks like a book lying on a shelf, with a colored spine ---------- */
    .book-card {
        display: flex;
        gap: 1rem;
        background: #FFFDF7;
        border: 1px solid rgba(107, 31, 31, 0.14);
        border-left: 8px solid var(--spine-color, #6B1F1F);
        border-radius: 4px 12px 12px 4px;
        padding: 1rem 1.1rem;
        margin-bottom: 1.1rem;
        box-shadow: 3px 4px 10px rgba(59, 42, 26, 0.10);
        animation: fadeSlideIn 0.45s ease-out both;
        transition: all 0.2s ease-in-out;
    }
    .book-card:hover {
        box-shadow: 4px 6px 16px rgba(59, 42, 26, 0.18);
        transform: translateY(-1px);
    }
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .book-card img {
        border-radius: 4px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.25);
        object-fit: cover;
    }

    /* Bookmark-ribbon rank badge */
    .book-rank {
        display: inline-block;
        position: relative;
        background: var(--spine-color, #6B1F1F);
        color: #FBF3E3;
        font-weight: 700;
        font-size: 0.72rem;
        padding: 2px 8px 2px 7px;
        margin-right: 8px;
        clip-path: polygon(0 0, 100% 0, 100% 100%, 50% 78%, 0 100%);
    }

    .book-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.12rem;
        font-weight: 700;
        color: #2B2118;
        margin-bottom: 0.15rem;
    }
    .book-author {
        font-size: 0.85rem;
        color: #8A7455;
        font-style: italic;
        margin-bottom: 0.5rem;
    }
    .book-desc {
        font-size: 0.85rem;
        color: #4A3E2C;
        line-height: 1.55;
    }
    .match-chip {
        display: inline-block;
        font-size: 0.66rem;
        font-weight: 700;
        color: #2F4B3C;
        background: rgba(47, 75, 60, 0.12);
        border: 1px solid rgba(47, 75, 60, 0.3);
        padding: 0.1rem 0.5rem;
        border-radius: 10px;
        margin-left: 0.4rem;
        vertical-align: middle;
        letter-spacing: 0.2px;
    }

    .stLinkButton > a {
        background: transparent !important;
        border: 1.5px solid #C9A227 !important;
        color: #6B1F1F !important;
        border-radius: 20px !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        font-family: 'Lora', serif !important;
    }
    .stLinkButton > a:hover {
        background: #C9A227 !important;
        color: #FBF3E3 !important;
        border-color: #C9A227 !important;
    }

    div[data-testid="stExpander"] {
        background: rgba(201, 162, 39, 0.06);
        border: 1px solid rgba(107, 31, 31, 0.15);
        border-radius: 8px;
    }

    hr { border-color: rgba(107, 31, 31, 0.15) !important; }

    /* ---------- Small flourish divider ---------- */
    .flourish {
        text-align: center;
        color: #C9A227;
        font-size: 1rem;
        letter-spacing: 0.6rem;
        opacity: 0.7;
        margin: 0.4rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Decorative shelf strip + Navbar ----------
st.markdown('<div class="shelf"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="navbar">
    <div class="navbar-logo">🖋️ Inkwell</div>
    <div class="navbar-tag">a shelf that reads you back</div>
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

    feature_names = vectorizer.get_feature_names_out()
    new_arr = new_tfidf.toarray()[0]

    keyword_list = []
    for idx in top_indices:
        book_arr = genre_tfidf[idx].toarray()[0]
        overlap = new_arr * book_arr
        top_terms_idx = overlap.argsort()[-3:][::-1]
        terms = [feature_names[t] for t in top_terms_idx if overlap[t] > 0]
        keyword_list.append(", ".join(terms) if terms else "general theme")

    results = genre_books.iloc[top_indices][['Title', 'Authors', 'Description']].reset_index(drop=True)
    results['Similarity'] = similarities[top_indices]
    results['Matched_Keywords'] = keyword_list
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
        image_url = f"https://placehold.co/90x120/6B1F1F/FBF3E3?text={initial}"

    shop_query = urllib.parse.quote(f"{title} {author}")
    amazon_link = f"https://www.amazon.in/s?k={shop_query}"

    return info_link, description, amazon_link, image_url


# ---------- Hero ----------
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">📚 Find Your Next Great Read</div>
    <div class="hero-subtitle">Describe a story you're in the mood for, and Inkwell will pull
    five handpicked titles from the shelves — matched by genre and content.</div>
    <div class="hero-rule"></div>
</div>
""", unsafe_allow_html=True)
with st.expander("📊 View Model Performance (Classification Report)"):
    st.markdown("**Overall test-set accuracy: 72%** across 20 genre classes (baseline: 5%)")
    perf_data = {
        "Genre": ["Cooking", "Fiction", "Computers", "Business & Economics", "Religion",
                  "Health & Fitness", "Juvenile Fiction", "Travel", "Political Science",
                  "Sports & Recreation", "Family & Relationships", "History",
                  "Biography & Autobiography", "Juvenile Nonfiction", "Science", "Humor",
                  "Reference", "Self-help", "Social Science", "Young Adult Fiction"],
        "Precision": [0.94, 0.94, 0.86, 0.83, 0.79, 0.78, 0.77, 0.70, 0.63, 0.61,
                      0.60, 0.52, 0.52, 0.53, 0.49, 0.46, 0.40, 0.45, 0.37, 0.33],
        "Recall": [0.94, 0.78, 0.82, 0.78, 0.70, 0.78, 0.69, 0.80, 0.71, 0.73,
                   0.69, 0.62, 0.57, 0.50, 0.69, 0.65, 0.55, 0.52, 0.40, 0.67],
        "F1-score": [0.94, 0.86, 0.84, 0.80, 0.74, 0.78, 0.73, 0.75, 0.67, 0.66,
                     0.64, 0.57, 0.54, 0.51, 0.57, 0.54, 0.46, 0.48, 0.39, 0.44],
    }
    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
    st.caption("Metrics computed on a held-out test set of 10,687 records during model training.")

user_input = st.text_area(
    "Book description",
    placeholder="e.g. A thrilling mystery about a detective solving a murder case in a small town...",
    height=120,
    label_visibility="collapsed",
)

_, btn_col, _ = st.columns([1, 1, 1])
with btn_col:
    get_recs = st.button("✨ Get Recommendations", type="primary", use_container_width=True)

SPINE_COLORS = ["#6B1F1F", "#2F4B3C", "#9C7A1A", "#4A3F6B", "#8B3A3A"]

if get_recs:
    if user_input.strip() == "":
        st.warning("Please enter a description first.")
    else:
        with st.spinner("🔎 Browsing the shelves for your next read..."):
            genre, recommendations = recommend_books(user_input, top_n=5)

        st.markdown(f"""
        <div style="text-align:center; margin: 1.2rem 0;">
            <span style="background: rgba(47,75,60,0.10); border: 1px solid #2F4B3C;
            color:#2F4B3C; padding: 0.45rem 1.2rem; border-radius: 20px; font-weight:600;
            font-size: 0.95rem; font-family:'Lora',serif;">🔖 Predicted Genre: {genre}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="flourish">❧ ❧ ❧</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">From the Shelf</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="results-meta">Showing <b>{len(recommendations)}</b> books matched to '
            f'<b>&ldquo;{genre}&rdquo;</b></div>',
            unsafe_allow_html=True,
        )

        for i, row in recommendations.iterrows():
            book_url, description, amazon_link, image_url = get_book_details(
                row['Title'], row['Authors'], row['Description']
            )

            short_desc = description if description and len(description) <= 220 else (description[:220] + "…" if description else "")
            delay = i * 0.08
            spine = SPINE_COLORS[i % len(SPINE_COLORS)]

            st.markdown(
                f'<div class="book-card" style="animation-delay:{delay}s; --spine-color:{spine};">',
                unsafe_allow_html=True,
            )
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
                <div class="book-title"><span class="book-rank" style="--spine-color:{spine};">{i+1}</span>
                <a href="{book_url}" target="_blank" style="color:#2B2118; text-decoration:none;">{row['Title']}</a>
                <span class="match-chip">{row['Similarity']*100:.0f}% text similarity</span>
                <div class="book-author">by {row['Authors']}</div>
                 f"<div style='font-size:0.78rem;color:#8A7455;margin-bottom:0.3rem;'>"
                 f"<em>Matched on:</em> {row['Matched_Keywords']}</div>",
                """, unsafe_allow_html=True)

                if short_desc:
                    st.markdown(f'<div class="book-desc">{short_desc}</div>', unsafe_allow_html=True)
                    if description and len(description) > 220:
                        with st.expander("Read full description"):
                            st.write(description)

                st.link_button("🛒 Buy on Amazon", amazon_link, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)
