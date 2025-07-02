import os
import chromadb
from chromadb.config import Settings

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(ROOT_PATH, "chroma_db")

client = chromadb.Client(Settings(persist_directory=DB_PATH, is_persistent=True))

print("\n✅ Collections in DB:")
for collection in client.list_collections():
    print(f"- Name: {collection.name}")

    # try to list some docs
    docs = collection.get(limit=5)
    print(f"  -> Documents found: {len(docs['documents'])}")
    for doc in docs['documents']:
        print(f"    - {doc[:50]}...")
    print()
