import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "movies_with_embeddings.csv"

if not DATA_PATH.exists():
    raise RuntimeError(f"Archivo no encontrado en {DATA_PATH}")

movies_df = pd.read_csv(DATA_PATH)
movies_df['embedding'] = movies_df['embedding'].apply(lambda x: list(map(float, x.strip('[]').split(','))))
embeddings = np.array(movies_df['embedding'].tolist())
movie_to_idx = {row['movieId']: idx for idx, row in movies_df.iterrows()}

app = FastAPI(title="Movie Recommender", description="Recomendaciones por similitud de coseno")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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
    query_embedding = embeddings[idx].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    sim_scores = list(enumerate(similarities))
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
