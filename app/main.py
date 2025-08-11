import faiss
import numpy as np
import pandas as pd
import uvicorn
import os
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "Online-Library-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Load the books dataset
df = pd.read_csv('assets/datasets/preprocessed.csv')

# Load the books embeddings
embeddings = np.load("assets/models/book_embeddings.npy")

# Load faiss index
index = faiss.read_index("assets/models/book_index.faiss")

# Initiating SentenceTransformer model
print("Iniating SentenceTransformer model, this may take a minute")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create FastAPI instance
app = FastAPI()

class BookRecommendRequest(BaseModel):
   index: int
   k: int=5


class BookRecommendResponse(BaseModel):
   recommendations: List[dict]


class BookRecommendRagRequest(BaseModel):
    query: str
    k: int=5


async def get_api_key(api_key: Optional[str] = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key",
        headers={"WWW-Authenticate": "Bearer"},
    )


# Main API gateway
@app.get('/')
async def health_check():
    return {'health_check': 'OK'}


@app.post('/recommend_rag', response_model=BookRecommendResponse, dependencies=[Depends(get_api_key)])
async def recommend_rag(payload: BookRecommendRagRequest):
    """
    Recommend books based on a query string.

    Args:
        query (str): The query string to search for.
        k (int): The number of recommendations to return.
    
    Returns:
        list[dict]: A list of dictionaries containing book recommendations.
    """

    # Encoding query into embedding
    query_embedding = model.encode(payload.query)

    # Search the FAISS index
    distances, indices = index.search(np.array([query_embedding]), payload.k + 1)

    # Return results as list of dictionaries
    return BookRecommendResponse(recommendations=df.iloc[indices[0][1:]].to_dict(orient='records'))



# Recommend API gateway
@app.post('/recommend', response_model=BookRecommendResponse, dependencies=[Depends(get_api_key)])
async def recommend(payload: BookRecommendRequest):
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