import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path

app = FastAPI(title="Movie Recommender", description="Recomendaciones por similitud de coseno")

MOVIES_PATH = Path("data/processed/movies_with_embeddings.csv")
SIMILARITY_PATH = Path("data/processed/movie_similarity.pkl")

if not MOVIES_PATH.exists() or not SIMILARITY_PATH.exists():
    raise RuntimeError("Ejecuta primero scripts/preprocess.py")

movies_df = pd.read_csv(MOVIES_PATH)
with open(SIMILARITY_PATH, 'rb') as f:
    similarity_matrix = pickle.load(f)

movie_to_idx = {row['movieId']: idx for idx, row in movies_df.iterrows()}

class RecommendRequest(BaseModel):
    movie_id: int
    top_k: int = 5

class RecommendResponse(BaseModel):
    movie_id: int
    title: str
    similarity_score: float

@app.get("/")
def root():
    return {"message": "Movie Recommender API"}

@app.post("/recommend", response_model=List[RecommendResponse])
def recommend(req: RecommendRequest):
    if req.movie_id not in movie_to_idx:
        raise HTTPException(status_code=404, detail="Movie ID not found")
    idx = movie_to_idx[req.movie_id]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:req.top_k]
    result = []
    for i, score in sim_scores:
        movie_row = movies_df.iloc[i]
        result.append(RecommendResponse(
            movie_id=int(movie_row['movieId']),
            title=movie_row['title'],
            similarity_score=float(score)
        ))
    return result
