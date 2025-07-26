import pickle
from fastapi import FastAPI

# Loading Pickle objects
with open(r'models\tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

with open(r'models\model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(r'models\index2title.pkl', 'rb') as f:
    index2title = pickle.load(f)

with open(r'models\title2index.pkl', 'rb') as f:
    title2index = pickle.load(f)

# Create FastAPI instance
app = FastAPI()

# Main API gateway
@app.get('/')
def health_check():
    return {'health_check': 'OK'}

# Recommend API gateway
@app.get('/recommend')
def recommend(title: str):
    """
    Recommends 5 books' titles related to given title.
    Args:
        title (str): Title of the book to find recommendations for.
        model: NearestNeighbors model used for recommendations.
    Returns:
        list: List of recommended book titles.
    """
    # If book not in the dataset, return error
    if not title.lower() in title2index.index:
        return None
    
    # Fast index lookup from title2index map
    idx = title2index[title.lower()]

    # Finding k closest indices in the high-dimensional vector space
    _, indices = model.kneighbors(tfidf_matrix[idx])

    # Excluding the first index since it's the given book itself
    indices = indices[0][1:]

    # Return the result as a Python list
    result = index2title.iloc[indices].values.tolist()
    return {"result": result}   