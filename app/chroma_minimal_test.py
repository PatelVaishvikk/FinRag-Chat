import os
import chromadb
from chromadb.config import Settings

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(ROOT_PATH, "chroma_db")

if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

print("Using ChromaDB path:", DB_PATH)

chroma_client = chromadb.Client(Settings(
    persist_directory=DB_PATH
))
coll = chroma_client.get_or_create_collection(name="test_collection")
coll.add(documents=["hello world"], embeddings=[[0.1]*384], ids=["test1"], metadatas=[{"foo": "bar"}])
print("Collections after add:", chroma_client.list_collections())
