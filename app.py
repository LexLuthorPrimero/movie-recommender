import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Recomendador de Películas")

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "processed" / "movies_with_embeddings.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['embedding'] = df['embedding'].apply(lambda x: list(map(float, x.strip('[]').split(','))))
    embeddings = np.array(df['embedding'].tolist())
    return df, embeddings

movies_df, embeddings = load_data()
movie_titles = movies_df.set_index('movieId')['title'].to_dict()

movie_id = st.selectbox("Selecciona una película", options=list(movie_titles.keys()), format_func=lambda x: movie_titles[x])
top_k = st.slider("Número de recomendaciones", 1, 10, 5)

if st.button("Recomendar"):
    idx = movies_df[movies_df['movieId'] == movie_id].index[0]
    query_embedding = embeddings[idx].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    sim_scores = list(enumerate(similarities))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_k]
    st.subheader("Películas recomendadas:")
    for i, score in sim_scores:
        title = movies_df.iloc[i]['title']
        st.write(f"- **{title}** (similitud: {score:.3f})")
