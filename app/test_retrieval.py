import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ---- Absolute Path for DB ----
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))

chroma_client = chromadb.Client(Settings(
    persist_directory=DB_PATH,
    is_persistent=True
))
print("DB_PATH:", DB_PATH)
print("Collections in DB:", chroma_client.list_collections())

# SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed(texts):
    return sbert_model.encode(texts).tolist()

# Example: Retrieve from 'marketing_docs'
try:
    collection = chroma_client.get_collection("marketing_docs")
except Exception as e:
    print("Error loading collection:", e)
    exit()

query = "Show me the latest marketing campaign performance"
query_vector = embed([query])[0]

results = collection.query(
    query_embeddings=[query_vector],
    n_results=3,
    include=["documents", "metadatas"]
)

for i, doc in enumerate(results['documents'][0]):
    print(f"Result {i+1}:")
    print("Document Chunk:", doc)
    print("Metadata:", results['metadatas'][0][i])
    print("-" * 60)
