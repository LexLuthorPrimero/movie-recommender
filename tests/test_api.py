import pytest
import pickle
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Movie Recommender API"}

def test_recommend_valid_with_mock():
    # Crear datos simulados
    mock_movies_df = pd.DataFrame({
        'movieId': [1, 2, 3],
        'title': ['Toy Story', 'Jumanji', 'Grumpier Old Men']
    })
    mock_similarity = np.array([[1.0, 0.8, 0.6],
                                 [0.8, 1.0, 0.5],
                                 [0.6, 0.5, 1.0]])

    with patch('src.api.movies_df', mock_movies_df), \
         patch('src.api.similarity_matrix', mock_similarity), \
         patch('src.api.movie_to_idx', {1: 0, 2: 1, 3: 2}):
        payload = {"movie_id": 1, "top_k": 2}
        response = client.post("/recommend", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]['title'] == 'Jumanji'  # segundo más similar
        assert data[0]['similarity_score'] == 0.8

def test_recommend_invalid():
    payload = {"movie_id": 999999, "top_k": 3}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 404
    assert response.json()["detail"] == "Movie ID not found"
