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
    # Verificar valores
    assert result.iloc[0]['Action'] == 1
    assert result.iloc[0]['Adventure'] == 1
    assert result.iloc[1]['Comedy'] == 1
    assert result.iloc[2]['Drama'] == 1

def test_create_embeddings_shape():
    movies = pd.DataFrame({'movieId': [1,2], 'genres': ['Action', 'Comedy']})
    ratings = pd.DataFrame({'movieId': [1,2], 'avg_rating': [4.5, 3.0]})
    embeddings = create_embeddings(movies, ratings)
    # Esperamos 2 géneros + 1 rating = 3 columnas
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 2 + 1
    # Verificar que el rating está normalizado (máximo 1)
    assert embeddings[0, -1] == 1.0  # rating 4.5 / max_rating (4.5)
    assert embeddings[1, -1] == 3.0/4.5
