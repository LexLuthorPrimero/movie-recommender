import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/ml-latest-small")
MOVIES_PATH = DATA_DIR / "movies.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"
OUTPUT_MOVIES = Path("data/processed/movies_with_embeddings.csv")
OUTPUT_SIMILARITY = Path("data/processed/movie_similarity.pkl")

def load_movies():
    df = pd.read_csv(MOVIES_PATH)
    logger.info(f"Películas cargadas: {len(df)}")
    return df

def load_ratings():
    df = pd.read_csv(RATINGS_PATH)
    avg_ratings = df.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings.rename(columns={'rating': 'avg_rating'}, inplace=True)
    logger.info(f"Ratings procesados: {len(avg_ratings)} películas")
    return avg_ratings

def encode_genres(movies_df):
    genres_series = movies_df['genres'].str.split('|')
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(genres_series)
    return pd.DataFrame(genres_encoded, columns=mlb.classes_, index=movies_df.index)

def create_embeddings(movies_df, ratings_df):
    genres_oh = encode_genres(movies_df).values
    movies_with_ratings = movies_df.merge(ratings_df, on='movieId', how='left')
    avg_ratings = movies_with_ratings['avg_rating'].fillna(0).values.reshape(-1, 1)
    max_rating = avg_ratings.max() if avg_ratings.max() > 0 else 1
    avg_ratings_norm = avg_ratings / max_rating
    embeddings = np.hstack([genres_oh, avg_ratings_norm])
    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings

def compute_similarity(embeddings):
    similarity = cosine_similarity(embeddings)
    logger.info(f"Matriz de similitud shape: {similarity.shape}")
    return similarity

def save_results(movies_df, embeddings, similarity):
    OUTPUT_MOVIES.parent.mkdir(parents=True, exist_ok=True)
    movies_with_emb = movies_df.copy()
    movies_with_emb['embedding'] = embeddings.tolist()
    movies_with_emb.to_csv(OUTPUT_MOVIES, index=False)
    import pickle
    with open(OUTPUT_SIMILARITY, 'wb') as f:
        pickle.dump(similarity, f)
    logger.info("Archivos guardados")

def main():
    movies = load_movies()
    ratings = load_ratings()
    embeddings = create_embeddings(movies, ratings)
    similarity = compute_similarity(embeddings)
    save_results(movies, embeddings, similarity)

if __name__ == "__main__":
    main()
