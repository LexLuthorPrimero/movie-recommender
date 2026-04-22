import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Recomendador de Películas")

movies_df = pd.read_csv("data/processed/movies_with_embeddings.csv")
movie_titles = movies_df.set_index('movieId')['title'].to_dict()

movie_id = st.selectbox("Selecciona una película", options=list(movie_titles.keys()), format_func=lambda x: movie_titles[x])
top_k = st.slider("Número de recomendaciones", 1, 10, 5)

API_URL = st.secrets.get("API_URL", "http://localhost:8000/recommend")

if st.button("Recomendar"):
    try:
        response = requests.post(API_URL, json={"movie_id": movie_id, "top_k": top_k})
        if response.status_code == 200:
            recs = response.json()
            st.subheader("Películas recomendadas:")
            for rec in recs:
                st.write(f"- **{rec['title']}** (similitud: {rec['similarity_score']:.3f})")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"No se pudo conectar a la API: {e}")
