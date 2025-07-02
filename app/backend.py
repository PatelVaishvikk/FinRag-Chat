import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai

# ---- Settings ----
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
EMBED_MODEL = 'all-MiniLM-L6-v2'
OPENAI_MODEL = "gpt-3.5-turbo"  # Or "gpt-4" if you have access

openai.api_key = os.getenv("OPENAI_API_KEY")

# ---- RBAC ----
USERS = {
    "alice": "finance",
    "bob": "marketing",
    "charlie": "hr",
    "dave": "engineering",
    "ceo": "c_level",
    "eve": "employee"
}
ROLE_COLLECTIONS = {
    "finance": ["finance_docs"],
    "marketing": ["marketing_docs"],
    "hr": ["hr_docs"],
    "engineering": ["engineering_docs"],
    "c_level": ["finance_docs", "marketing_docs", "hr_docs", "engineering_docs", "general_docs"],
    "employee": ["general_docs"]
}

def authenticate(username):
    return USERS.get(username)

def get_collections_for_role(role):
    return ROLE_COLLECTIONS.get(role, [])

# ---- ChromaDB ----
client = chromadb.Client(Settings(
    persist_directory=DB_PATH,
    is_persistent=True
))
embedder = SentenceTransformer(EMBED_MODEL)

def embed(texts):
    return embedder.encode(texts).tolist()

def search_collections(collection_names, query, top_k=8, min_score=0.08):
    query_vec = embed([query])[0]
    results = []
    for cname in collection_names:
        try:
            collection = client.get_collection(cname)
        except Exception:
            continue
        res = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            similarity = 1 - dist
            if similarity >= min_score:
                results.append({
                    "document": doc,
                    "metadata": meta,
                    "collection": cname,
                    "similarity": similarity
                })
    return sorted(results, key=lambda x: -x["similarity"])[:top_k]

# ---- FastAPI Setup ----
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo/testing. Restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AuthRequest(BaseModel):
    username: str

class AskRequest(BaseModel):
    username: str
    question: str

@app.post("/auth")
def login(req: AuthRequest):
    role = authenticate(req.username)
    if not role:
        raise HTTPException(status_code=403, detail="Invalid user")
    return {"username": req.username, "role": role}

@app.post("/ask")
def ask(req: AskRequest):
    role = authenticate(req.username)
    if not role:
        raise HTTPException(status_code=403, detail="Invalid user")
    collections = get_collections_for_role(role)
    if not collections:
        raise HTTPException(status_code=403, detail="No collections for this role")
    results = search_collections(collections, req.question)
    if not results:
        return {"answer": None, "sources": [], "context": ""}
    # Use the *most relevant* results as context for LLM
    context = "\n\n".join([f"From {r['metadata']['source']}:\n{r['document']}" for r in results])
    # Prompt for LLM
    prompt = (
        f"You are FinSolve's internal assistant. Use only the provided CONTEXT to answer. "
        f"Include the source file name in your answer if relevant. If the answer isn't present, say 'No relevant answer found.'\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {req.question}\n\n"
        f"ANSWER:"
    )
    # Call OpenAI LLM
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for a FinTech company."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=512
    )
    answer = response['choices'][0]['message']['content']
    return {
        "answer": answer,
        "sources": [{"source": r['metadata']['source'], "similarity": r['similarity']} for r in results],
        "context": context
    }

@app.get("/")
def root():
    return {"msg": "FinSolve RAG + RBAC Backend running."}
