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

    results = genre_books.iloc[top_indices][['Title', 'Authors']].reset_index(drop=True)
    return predicted_genre, results

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
            st.markdown(f"**{i+1}. {row['Title']}**  \n*by {row['Authors']}*")

st.markdown("---")
st.caption("M.Sc. Statistics Project — Genre classification (TF-IDF + Logistic Regression) with content-based recommendation (cosine similarity).")
