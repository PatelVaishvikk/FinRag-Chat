import os
import glob
import toml
import csv
import re
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# ========= CONFIG =========
# Chunk size for splitting documents into smaller pieces for embedding
CHUNK_SIZE = 1000
# Overlap between chunks to maintain context across splits
OVERLAP = 200
# Maximum chunk size specifically for CSV rows, to avoid overly long single embeddings
CSV_MAX_CHUNK = 1200

# Define paths relative to the script's location
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(ROOT_PATH, "chroma_db")
DATA_PATH = os.path.join(ROOT_PATH, "data")
# Create the ChromaDB directory if it doesn't exist
os.makedirs(DB_PATH, exist_ok=True)

# List of departments, corresponding to data subfolders and ChromaDB collection names
departments = ["engineering", "finance", "marketing", "hr", "general"]

# ========= LOAD API KEY =========
# Attempt to load OpenAI API key from environment variables or Streamlit secrets file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    secrets_path = os.path.join(ROOT_PATH, "frontend/.streamlit/secrets.toml")
    if os.path.exists(secrets_path):
        secrets = toml.load(secrets_path)
        OPENAI_API_KEY = secrets.get("OPENAI_API_KEY")

# Validate that the API key is present and appears to be in the correct format
if not OPENAI_API_KEY or not OPENAI_API_KEY.strip().startswith("sk-"):
    raise ValueError(
        "❌ OPENAI_API_KEY not found or invalid. Set it in environment or frontend/.streamlit/secrets.toml"
    )

# Initialize OpenAI client and define the embedding model
client_oai = OpenAI(api_key=OPENAI_API_KEY)
EMBEDDING_MODEL = "text-embedding-3-small"  # ENSURE THIS MATCHES THE CHATBOT

print(f"✅ Using embedding model: {EMBEDDING_MODEL}")

# ========= EMBEDDING =========
# Function to generate embeddings for a list of texts using the specified OpenAI model
def embed(texts):
    try:
        response = client_oai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        embeddings = [d.embedding for d in response.data]
        print(f"✅ Generated {len(embeddings)} embeddings with {len(embeddings[0])} dimensions")
        return embeddings
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        raise

# ========= VECTOR DB =========
# Initialize ChromaDB client in persistent mode with explicit settings
chroma_client = chromadb.Client(Settings(
    persist_directory=DB_PATH,
    is_persistent=True,
    anonymized_telemetry=False
))

# ========= UTILITY FUNCTIONS =========
# Chunks text into sentences, ensuring chunks are within CHUNK_SIZE with OVERLAP
def chunk_sentences(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    sentences = re.split(r'(?<=[.!?]) +', text) # Split by common sentence terminators
    chunks = []
    buffer = ""
    for i, sentence in enumerate(sentences):
        # If adding the current sentence keeps the buffer within chunk_size, add it
        if len(buffer) + len(sentence) < chunk_size:
            buffer += " " + sentence
        else:
            # If buffer is too long, save the current buffer as a chunk
            if buffer.strip():
                chunks.append(buffer.strip())
            # Start a new buffer, including an overlap of previous sentences
            buffer = " ".join(sentences[max(0, i-2):i+1]) # Overlap last 2 sentences + current
    # Add any remaining text in the buffer as the last chunk
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks

# Chunks long lines of text (e.g., from CSV) into fixed-size segments
def chunk_long_text(text, max_length=CSV_MAX_CHUNK):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# ========= CLEAR EXISTING DATA =========
print("🧹 Clearing existing collections...")
for dept in departments:
    try:
        collection = chroma_client.get_collection(f"{dept}_docs")
        chroma_client.delete_collection(f"{dept}_docs")
        print(f"  ✅ Cleared existing {dept}_docs collection")
    except Exception:
        print(f"  ℹ️  No existing {dept}_docs collection to clear")

# ========= INGESTION LOOP =========
# Iterate through each department to ingest its documents
for dept in departments:
    folder = os.path.join(DATA_PATH, dept)
    if not os.path.exists(folder):
        print(f"⚠️  Folder missing for department: {dept}. Skipping.")
        continue

    # Get or create a collection in ChromaDB for the current department
    collection = chroma_client.get_or_create_collection(
        name=f"{dept}_docs",
        metadata={"hnsw:space": "cosine"}  # Explicitly set cosine distance
    )
    print(f"Processing department: {dept}")

    # ✅ 1. Ingest Markdown files
    md_files = glob.glob(os.path.join(folder, "*.md"))
    if md_files:
        for path in md_files:
            print(f"  Ingesting Markdown: {os.path.basename(path)}")
            # FIXED: encoding="utf-8" instead of encoding="utf="8"
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                print(f"    ⚠️  Empty file skipped: {path}")
                continue

            # Split Markdown content by headings and then chunk sentences within sections
            sections = re.split(r'(?=^#{1,6} )', content, flags=re.MULTILINE)
            chunks = []
            for section in sections:
                if section.strip():  # Only process non-empty sections
                    chunks.extend(chunk_sentences(section.strip()))

            if not chunks:
                print(f"    ⚠️  No chunks generated from: {path}")
                continue

            print(f"    ✅ {os.path.basename(path)} → {len(chunks)} chunks (Markdown)")
            # Generate embeddings for the chunks
            embeddings = embed(chunks)
            # Add chunks, embeddings, metadata, and unique IDs to the ChromaDB collection
            collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=[{"department": dept, "source": os.path.basename(path), "type": "markdown"} for _ in chunks],
                ids=[f"{dept}-{os.path.basename(path)}-md-{i}" for i in range(len(chunks))]
            )
    else:
        print(f"  ⚠️  No markdown files found in {dept}")

    # ✅ 2. Ingest CSV files
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    if csv_files:
        for path in csv_files:
            print(f"  Ingesting CSV: {os.path.basename(path)}")
            # FIXED: encoding="utf-8" instead of encoding="utf="8"
            with open(path, "r", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                rows = []
                header_row = None
                for i, row in enumerate(reader):
                    # Capture header row
                    if i == 0:
                        header_row = row
                        # Skip if it looks like a header
                        if all(re.match(r'^\s*[\w\s]+\s*$', col) for col in row):
                            continue
                    # Join non-empty columns with " | "
                    joined = " | ".join(col.strip() for col in row if col.strip())
                    if joined:
                        rows.append(joined)

            if not rows:
                print(f"    ⚠️  Empty CSV skipped: {path}")
                continue

            chunks = []
            for row in rows:
                # If a CSV row is very long, chunk it further
                if len(row) > CSV_MAX_CHUNK:
                    chunks.extend(chunk_long_text(row, CSV_MAX_CHUNK))
                else:
                    chunks.append(row)

            print(f"    ✅ {os.path.basename(path)} → {len(chunks)} chunks (CSV)")
            embeddings = embed(chunks)
            collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=[{"department": dept, "source": os.path.basename(path), "type": "csv"} for _ in chunks],
                ids=[f"{dept}-{os.path.basename(path)}-csv-{i}" for i in range(len(chunks))]
            )
    else:
        print(f"  ⚠️  No CSV files found in {dept}")

    # Verify collection after ingestion
    final_count = collection.count()
    print(f"✅ Finished ingesting all documents for {dept}. Total documents: {final_count}\n")

# ========= FINAL VERIFICATION =========
print("🔍 Final verification:")
collections = chroma_client.list_collections()
for col in collections:
    count = col.count()
    print(f"  - {col.name}: {count} documents")

print(f"✅ ALL DOCUMENTS INGESTED to {DB_PATH}")
print(f"✅ Using embedding model: {EMBEDDING_MODEL} (ensure chatbot uses the same!)")