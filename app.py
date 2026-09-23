import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Page setup ----------
st.set_page_config(page_title="Book Recommendation System", page_icon="📚", layout="centered")

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
    except Exception:
        pass

    if not info_link:
        try:
            query = urllib.parse.quote(f"{title} {author}")
            url = f"https://openlibrary.org/search.json?q={query}&limit=1"
            response = requests.get(url, timeout=3)
            data = response.json()
            if data.get("docs"):
                key = data["docs"][0].get("key")
                if key:
                    info_link = f"https://openlibrary.org{key}"
        except Exception:
            pass

    if not info_link:
        search_query = urllib.parse.quote(f"{title} {author}")
        info_link = f"https://www.google.com/search?tbm=bks&q={search_query}"

    shop_query = urllib.parse.quote(f"{title} {author}")
    amazon_link = f"https://www.amazon.in/s?k={shop_query}"

    return info_link, description, amazon_link
    # ---------- UI ----------
st.title("📚 Book Recommendation System")
st.write("Enter a book description or the kind of story you're looking for, and get 5 similar book recommendations based on genre and content.")

user_input = st.text_area(
    "Book description",
    placeholder="e.g. A thrilling mystery about a detective solving a murder case in a small town...",
    height=120,
)

if st.button("Get Recommendations", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter a description first.")
    else:
        with st.spinner("Analyzing description..."):
            genre, recommendations = recommend_books(user_input, top_n=5)

        st.success(f"**Predicted Genre:** {genre}")
        st.subheader("Top 5 Recommended Books")

        for i, row in recommendations.iterrows():
            book_url, description, amazon_link = get_book_details(
                row['Title'], row['Authors'], row['Description']
            )
            st.markdown(f"**{i+1}. [{row['Title']}]({book_url})**  \n*by {row['Authors']}*")
    
            if description:
                if len(description) > 200:
                    st.caption(description[:200] + "...")
                    with st.expander("Read full description"):
                        st.write(description)
                else:
                    st.caption(description)
    
            st.link_button("🛒 Buy on Amazon", amazon_link, use_container_width=True)
            st.markdown("---")
