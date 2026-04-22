import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Movie Recommender API"}

def test_recommend_valid():
    payload = {"movie_id": 1, "top_k": 3}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    assert "title" in data[0]
    assert "similarity_score" in data[0]

def test_recommend_invalid():
    payload = {"movie_id": 999999, "top_k": 3}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 404

def test_recommend_default_top_k():
    payload = {"movie_id": 1}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    assert len(response.json()) == 5
