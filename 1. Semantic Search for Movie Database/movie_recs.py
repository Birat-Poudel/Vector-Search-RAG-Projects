from urllib.parse import quote_plus
import pymongo
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Load MongoDB credentials
username = quote_plus(os.getenv("MONGODB_USERNAME"))
password = quote_plus(os.getenv("MONGODB_PASSWORD"))

# Construct MongoDB URI
uri = f"mongodb+srv://{username}:{password}@cluster0.ukyvufh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Initialize MongoDB client
client = pymongo.MongoClient(uri)
db = client.sample_mflix
collection = db.movies

# Load Hugging Face token
hf_token = os.getenv("HF_TOKEN")

# Hugging Face API endpoint for embeddings
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def generate_embedding(text: str) -> list[float]:

  response = requests.post(
    embedding_url,
    headers={"Authorization": f"Bearer {hf_token}"},
    json={"inputs": text})

  if response.status_code != 200:
    raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")

  return response.json()

# Uncomment to generate embeddings for existing MongoDB documents
# for doc in collection.find({'plot':{"$exists": True}}).limit(50):
#   doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
#   collection.replace_one({'_id': doc['_id']}, doc)

# Query for semantic search
query = "imaginary characters from outer space at war"

# Perform the vector search
results = collection.aggregate([
  {"$vectorSearch": {
    "queryVector": generate_embedding(query),
    "path": "plot_embedding_hf",
    "numCandidates": 100,
    "limit": 4,
    "index": "PlotSemanticSearch",
      }}
])

# Print the results
for document in results:
    print(f'Movie Name: {document["title"]},\nMovie Plot: {document["plot"]}\n')