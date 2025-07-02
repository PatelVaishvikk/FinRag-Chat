import os, sys
import streamlit as st

# ─── Make `app/` importable ─────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ─── Authentication helpers ──────────────────────────────────────────────────
# auth.py defines:
#   _CREDENTIALS = { username: (password, role), ... }
#   def authenticate(username, password) -> role or None
from app.auth import authenticate, _CREDENTIALS as CREDENTIALS

# ─── RBAC CONFIG ─────────────────────────────────────────────────────────────
ROLE_COLLECTIONS = {
    "finance":     ["finance_docs"],
    "marketing":   ["marketing_docs"],
    "hr":          ["hr_docs"],
    "engineering": ["engineering_docs"],
    "c_level":     ["finance_docs","marketing_docs","hr_docs","engineering_docs","general_docs"],
    "employee":    ["general_docs"]
}

def collections_for(role: str):
    return ROLE_COLLECTIONS.get(role, [])

# ========= PAGE CONFIG =========
st.set_page_config(page_title="Fintech Insight Architect", layout="wide")

# ========= LOGIN UI =========
if not st.session_state.get("authenticated", False):
    st.title("🔒 FinTech Insight Architect Login")

    username = st.text_input("Username", placeholder="e.g. alice, bob, charlie")
    password = st.text_input("Password", type="password")

    if st.button("Log In"):
        role = authenticate(username.strip().lower(), password)
        if role:
            st.session_state.authenticated = True
            st.session_state.username = username.strip().lower()
            st.session_state.role = role
            st.success(f"✅ Logged in as **{username}** (Role: {role})")
        else:
            st.error("❌ Invalid username or password. Try one of: " + ", ".join(CREDENTIALS.keys()))
    st.stop()

# ========= MAIN APP UI =========
st.title("Fintech Insight Architect")
allowed_cols = collections_for(st.session_state.role)
st.info(f"**Your access:** {', '.join(allowed_cols)}")
debug_mode = st.checkbox("🔧 Enable Debug Mode", value=False)

# ========= OPENAI & CHROMA SETUP =========
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import re

# OpenAI
try:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
    client_oai = OpenAI(api_key=OPENAI_API_KEY)
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHAT_COMPLETION_MODEL = "gpt-4o"
except Exception as e:
    st.error(f"OpenAI API setup failed: {e}")
    st.stop()

# ChromaDB
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../chroma_db"))
try:
    client_db = chromadb.Client(Settings(
        persist_directory=DB_PATH,
        is_persistent=True,
        anonymized_telemetry=False
    ))
    collection_names = [col.name for col in client_db.list_collections()]
    for name in collection_names:
        st.sidebar.write(f"- {name}")
except Exception as e:
    st.error(f"ChromaDB init failed: {e}")
    st.stop()

# Embedding cache
decorator = st.cache_resource
@decorator(show_spinner="🔎 Loading embeddings…")
def embed(texts):
    try:
        resp = client_oai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [d.embedding for d in resp.data]
    except Exception:
        return []

# ========= KEYWORD FALLBACK FUNCTIONS =========
def keyword_fallback(allowed_cols, query):
    query_lower = query.lower()
    keywords = query_lower.split()
    results = []
    for cname in allowed_cols:
        try:
            col = client_db.get_collection(cname)
            data = col.get(include=["documents","metadatas"])
            docs = data.get("documents", [])
            metas = data.get("metadatas", [{}]*len(docs))
            for doc, meta in zip(docs, metas):
                score = sum(1 for kw in keywords if kw in doc.lower()) / len(keywords)
                if score > 0.2:
                    results.append({ 'content': doc, 'score': score, 'source': cname, 'metadata': meta })
        except Exception as e:
            if debug_mode: st.error(f"Fallback error {cname}: {e}")
    return sorted(results, key=lambda x: x['score'], reverse=True)[:7]

def engineering_keyword_fallback(query):
    try:
        col = client_db.get_collection("engineering_docs")
        data = col.get(include=["documents","metadatas"])
        docs = data.get("documents", [])
        metas = data.get("metadatas", [{}]*len(docs))
        eng_keys = ["agile","methodology","scrum","technical","architecture","software","engineering"]
        out = []
        for doc, meta in zip(docs, metas):
            text = doc.lower()
            matches = sum(1 for k in eng_keys if k in text)
            if matches:
                score = min(matches*0.2+0.3, 1.0)
                out.append({ 'content': doc, 'score': score, 'source':'engineering_docs','metadata':meta })
        return sorted(out, key=lambda x: x['score'], reverse=True)[:5]
    except Exception:
        return []

# ========= RETRIEVE VECTOR HITS =========
def retrieve(allowed_cols, query, top_k=7):
    qvecs = embed([query])
    if not qvecs: return []
    qvec = qvecs[0]
    hits = []
    for cname in allowed_cols:
        try:
            col = client_db.get_collection(cname)
            count = col.count()
            if count == 0: continue
            res = col.query(query_embeddings=[qvec], n_results=min(top_k,count), include=["documents","distances","metadatas"])
            for doc,dist,meta in zip(res["documents"][0], res["distances"][0], res.get("metadatas")[0]):
                sim = 1 - min(max(dist,0),2)/2
                hits.append((doc,sim,cname,meta))
        except Exception as e:
            if debug_mode: st.error(f"Retrieve error {cname}: {e}")
    hits.sort(key=lambda x: x[1], reverse=True)
    # keep top
    filtered = []
    for doc,score,src,meta in hits:
        if score>=0.3 or len(filtered)<3:
            filtered.append((doc,score,src,meta))
    return [{ 'content':d,'score':s,'source':src,'metadata':m } for d,s,src,m in filtered[:top_k]]

# ========= RAG ANSWER =========
def rag_answer(question, docs):
    if not docs: return "I don't know based on the provided context."
    context = []
    sources = set()
    for i,doc in enumerate(docs[:5]):
        context.append(f"Doc{i+1}({doc['source']},{doc['score']:.2f}): {doc['content']}")
        sources.add(doc['source'])
    system = ("You are a FinTech assistant... only use provided context.")
    msg = [{"role":"system","content":system}, {"role":"user","content":f"Context:\n{'\n\n'.join(context)}\n\nQ: {question}"}]
    try:
        resp = client_oai.chat.completions.create(model=CHAT_COMPLETION_MODEL, messages=msg, temperature=0.3)
        ans = resp.choices[0].message.content.strip()
        if ans != "I don't know based on the provided context.":
            ans += f"\n\n**Sources:** {', '.join(sources)}"
        return ans
    except Exception as e:
        return f"Error: {e}"

# ========= CHAT UI =========
query = st.text_area("Ask a question:", height=100, placeholder="e.g., What are our marketing expenses?")
if st.button("Submit") and query.strip():
    with st.spinner("🔍 Retrieving…"):
        docs = retrieve(allowed_cols, query)
        if not docs or all(d['score']<0.4 for d in docs):
            st.warning("⚠️ Low similarity – trying keyword fallback")
            docs = keyword_fallback(allowed_cols, query) or docs
        if st.session_state.role=='engineering' and (not docs or all(d['score']<0.4 for d in docs)):
            docs = engineering_keyword_fallback(query) or docs
    if not docs:
        answer = "I don't know based on the provided context."
    else:
        answer = rag_answer(query, docs)
    st.markdown("### Answer")
    st.markdown("---")
    st.write(answer)

# ========= DIAGNOSTICS =========
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Diagnostics")
if st.sidebar.button("Test OpenAI Connection"):
    try:
        emb = embed(["test"])[0]
        st.sidebar.success("OpenAI OK – dim="+str(len(emb)))
    except Exception as e:
        st.sidebar.error(f"OpenAI error: {e}")

if st.sidebar.button("Check Collections"):
    st.sidebar.write(collection_names)

if st.sidebar.button("Check Distances") and "engineering_docs" in collection_names:
    test_e = embed(["agile methodology"])[0]
    res = client_db.get_collection("engineering_docs").query(query_embeddings=[test_e], n_results=3, include=["distances"])
    st.sidebar.write(res["distances"][0])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Notes")
st.sidebar.markdown("- Re-run ingestion after updates\n- Ensure same models in ingestion & app\n- Verify chroma_db structure")
