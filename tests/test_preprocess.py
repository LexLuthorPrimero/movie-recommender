import pytest
import pandas as pd
import numpy as np
from scripts.preprocess import encode_genres, create_embeddings

def test_encode_genres():
    df = pd.DataFrame({'genres': ['Action|Adventure', 'Comedy', 'Drama']})
    result = encode_genres(df)
    assert 'Action' in result.columns
    assert 'Adventure' in result.columns
    assert 'Comedy' in result.columns
    assert 'Drama' in result.columns
    assert result.shape[0] == 3

def test_create_embeddings_shape():
    movies = pd.DataFrame({'movieId': [1,2], 'genres': ['Action', 'Comedy']})
    ratings = pd.DataFrame({'movieId': [1,2], 'avg_rating': [4.5, 3.0]})
    embeddings = create_embeddings(movies, ratings)
    assert embeddings.shape[0] == 2
    # Número de columnas = número de géneros (en este caso 2) + 1 (rating)
    assert embeddings.shape[1] == 2 + 1
