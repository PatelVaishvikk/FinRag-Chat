import os
import glob
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(ROOT_PATH, "chroma_db")
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed(texts):
    return sbert_model.encode(texts).tolist()

chroma_client = chromadb.Client(Settings(
    persist_directory=DB_PATH,
    is_persistent=True
))

departments = ["engineering", "finance", "marketing", "hr", "general"]
base_path = os.path.abspath(os.path.join(ROOT_PATH, 'data'))

for dept in departments:
    doc_folder = os.path.join(base_path, dept)
    print(f"\n--- {dept.upper()} ---")
    print(f"Looking for .md files in: {doc_folder}")
    md_files = glob.glob(os.path.join(doc_folder, "*.md"))
    print(f"Found {len(md_files)} files: {md_files}")
    collection = chroma_client.get_or_create_collection(name=f"{dept}_docs")
    for path in md_files:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = [content[i:i+500] for i in range(0, len(content), 500)]
        embeddings = embed(chunks)
        collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"department": dept, "source": os.path.basename(path)} for _ in chunks],
            ids=[f"{dept}-{os.path.basename(path)}-{i}" for i in range(len(chunks))]
        )
        print(f"Ingested {len(chunks)} chunks from {os.path.basename(path)}.")
    print(f"Completed ingestion for {dept}.")

print(f"\nAll documents ingested successfully to {DB_PATH}!")
print("Collections after ingestion:", chroma_client.list_collections())
