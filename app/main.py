import faiss
import numpy as np
import pandas as pd
import uvicorn
import ast
from typing import List
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

# Load the books dataset
df = pd.read_csv(r'datasets\dataset.csv')

# Turning string columns into lists
df['author'] = df['author'].apply(lambda x: ast.literal_eval(x))
df['genre'] = df['genre'].apply(lambda x: ast.literal_eval(x))

# Load the books embeddings
embeddings = np.load(r"datasets\book_embeddings.npy")

# Load faiss index
index = faiss.read_index(r"datasets\book_index.faiss")

# Create FastAPI instance
app = FastAPI()

class BookRecommendRequest(BaseModel):
   index: int
   k: int=5


class BookRecommendResponse(BaseModel):
   recommendations: List[dict]


# Main API gateway
@app.get('/')
def health_check():
    return {'health_check': 'OK'}

# Recommend API gateway
@app.post('/recommend', response_model=BookRecommendResponse)
def recommend(payload: BookRecommendRequest):
  """
  Recommends k books similar to the given book.

  Args:
    book_index (int): The index of the book in the dataframe.
    k (int): The number of books to recommend.

  Returns:
    A List[dict] containing the recommended books.
  """

  # getting embedding vector by book_index
  query_vector = embeddings[payload.index].reshape(1, -1)

  # getting k + 1 similar vectors (+1 to exclude self)
  D, I = index.search(query_vector, payload.k + 1)

  # returning a dataframe with selected indices (slicing by [1:] to exclude self)
  # converting dataframe into json
  return BookRecommendResponse(recommendations=df.iloc[I[0][1:]].to_dict(orient='records'))


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)