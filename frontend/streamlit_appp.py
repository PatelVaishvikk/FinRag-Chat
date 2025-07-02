



import os, sys
import streamlit as st

# ─── Make `app/` importable ─────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ─── Static authentication import ─────────────────────────────────────────────
from app.auth import authenticate, USERS as AUTH_USERS

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fintech Insight Architect", layout="wide")

# ─── LOGIN UI ────────────────────────────────────────────────────────────────
st.title("🔒 Please log in to access FinTech Insights")
username = st.text_input("Username", placeholder="e.g. alice, bob, charlie")
password = st.text_input("Password", type="password")

# Halt here until both fields are filled
if not username or not password:
    st.info("🔑 Enter both username and password to continue.")
    st.stop()

# Authenticate (static lookup)
role = authenticate(username)
if not role:
    st.error(f"❌ Invalid credentials. Try one of: {', '.join(AUTH_USERS.keys())}")
    st.stop()

# Login successful
st.success(f"✅ Logged in as **{username}** (Role: {role})")

# ─── AFTER LOGIN: MAIN APP UI ─────────────────────────────────────────────────
# Now that the user is authenticated, show the main application
st.title("Fintech Insight Architect")

# Continue with your existing OpenAI, Chroma, and RAG pipeline below...

import chromadb
from chromadb.config import Settings
from openai import OpenAI
import re

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

# ─── OPENAI SETUP ─────────────────────────────────────────────────────────────
try:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
    client_oai = OpenAI(api_key=OPENAI_API_KEY)
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHAT_COMPLETION_MODEL = "gpt-4o"
except Exception as e:
    st.error(f"OpenAI API setup failed: {e}")
    st.stop()

# ─── VECTOR STORE SETUP ───────────────────────────────────────────────────────
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../chroma_db"))
try:
    client_db = chromadb.Client(Settings(
        persist_directory=DB_PATH,
        is_persistent=True,
        anonymized_telemetry=False
    ))
    # Sidebar: list collections
    for col in client_db.list_collections():
        st.sidebar.write(f"- {col.name}")
except Exception as e:
    st.error(f"ChromaDB init failed: {e}")
    st.stop()

# ─── EMBEDDING FUNCTION ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔎 Loading embeddings…")
def embed(texts):
    try:
        resp = client_oai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [d.embedding for d in resp.data]
    except Exception:
        return []



# ========== KEYWORD FALLBACK FUNCTIONS ==========
def keyword_fallback(allowed_cols, query):
    """Generic keyword-based document retrieval when embeddings fail"""
    query_lower = query.lower()
    keywords = query_lower.split()

    results = []

    for cname in allowed_cols:
        try:
            col = client_db.get_collection(cname)
            all_docs_result = col.get(include=["documents", "metadatas"])

            if all_docs_result and "documents" in all_docs_result:
                docs = all_docs_result["documents"]
                metadatas = all_docs_result.get("metadatas", [{}] * len(docs))

                for doc, metadata in zip(docs, metadatas):
                    doc_lower = doc.lower()
                    # Simple keyword matching with scoring
                    score = sum(1 for keyword in keywords if keyword in doc_lower) / len(keywords)

                    if score > 0.2:  # At least 20% keyword match
                        results.append({
                            'content': doc,
                            'score': score,
                            'source': cname,
                            'metadata': metadata or {}
                        })
        except Exception as e:
            st.error(f"Error in keyword fallback for {cname}: {e}")
            continue

    # Sort by score and return top results
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:7]


def engineering_keyword_fallback(query):
    """Engineering-specific keyword fallback"""
    try:
        col = client_db.get_collection("engineering_docs")
        all_docs_result = col.get(include=["documents", "metadatas"])

        if all_docs_result and "documents" in all_docs_result:
            docs = all_docs_result["documents"]
            metadatas = all_docs_result.get("metadatas", [{}] * len(docs))

            query_lower = query.lower()
            engineering_keywords = [
                "agile", "methodology", "scrum", "development", "technical",
                "architecture", "software", "engineering", "process", "code"
            ]

            results = []
            for doc, metadata in zip(docs, metadatas):
                doc_lower = doc.lower()
                # Check for engineering-specific keywords
                matches = sum(1 for keyword in engineering_keywords if keyword in doc_lower)
                if matches > 0 or any(word in doc_lower for word in query_lower.split()):
                    score = min(matches * 0.2 + 0.3, 1.0)  # Boost engineering content
                    results.append({
                        'content': doc,
                        'score': score,
                        'source': 'engineering_docs',
                        'metadata': metadata or {}
                    })

            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:5]

    except Exception as e:
        st.error(f"Error in engineering fallback: {e}")
    return []


# ========== RETRIEVE VECTOR HITS ==========
def retrieve(allowed_cols, query, top_k=7):
    st.write(f"**Debug Info:** Searching in collections: {allowed_cols}")

    qvec = embed([query])
    if not qvec:
        return []

    qvec = qvec[0]
    hits = []

    for cname in allowed_cols:
        try:
            col = client_db.get_collection(cname)

            # Check if collection has documents
            collection_count = col.count()
            st.write(f"**Debug:** Collection '{cname}' has {collection_count} documents")

            if collection_count == 0:
                st.warning(f"Collection '{cname}' is empty!")
                continue

            res = col.query(
                query_embeddings=[qvec],
                n_results=min(top_k, collection_count),
                include=["documents", "distances", "metadatas"]
            )

            st.write(f"**Debug:** Retrieved {len(res['documents'][0])} documents from '{cname}'")

            # Handle ChromaDB distance calculation properly
            for i, (doc, dist) in enumerate(zip(res["documents"][0], res["distances"][0])):
                metadata = res.get("metadatas", [[{}]])[0][i] if res.get("metadatas") else {}

                # Calculate similarity score from cosine distance
                # Cosine distance should be between 0 and 2, with 0 being identical
                if dist < 0:
                    similarity_score = 1.0  # Identical (shouldn't happen)
                elif dist > 2:
                    similarity_score = 0.0  # Completely different
                else:
                    similarity_score = 1 - (dist / 2)  # Convert to 0-1 similarity

                hits.append((doc, similarity_score, cname, metadata))
                st.write(f"**Debug:** Doc {i + 1} distance: {dist:.3f}, similarity: {similarity_score:.3f}")

        except Exception as e:
            st.error(f"Error retrieving from collection {cname}: {e}")
            continue

    # Sort by similarity score (higher is better)
    hits.sort(key=lambda x: x[1], reverse=True)

    # Filter hits based on similarity threshold
    filtered_hits = []
    similarity_threshold = 0.3  # Reasonable threshold for text similarity

    for doc, score, source, metadata in hits:
        if score >= similarity_threshold or len(filtered_hits) < 3:  # Always keep top 3
            filtered_hits.append((doc, score, source, metadata))

    # Convert to document objects
    result_docs = []
    for doc, score, source, metadata in filtered_hits[:top_k]:
        result_docs.append({
            'content': doc,
            'score': score,
            'source': source,
            'metadata': metadata
        })

    # st.write(f"**Debug:** Total hits: {len(hits)}, Filtered: {len(filtered_hits)}, Returning: {len(result_docs)}")

    return result_docs


# ========== RAG ANSWER ==========
def rag_answer(question, doc_objects):
    if not doc_objects:
        return "I don't know based on the provided context."

    # Extract content and create context with source attribution
    context_parts = []
    sources = []

    for i, doc_obj in enumerate(doc_objects[:5]):  # Limit to top 5 for context window
        content = doc_obj['content']
        source = doc_obj['source']
        score = doc_obj['score']

        context_parts.append(f"Document {i + 1} (Source: {source}, Relevance: {score:.3f}):\n{content}")
        sources.append(source)

    context = "\n\n".join(context_parts)
    unique_sources = list(set(sources))

    system = (
        "You are a highly capable FinTech assistant for FinSolve Technologies. "
        "You are provided with relevant documents from the company's knowledge base. "
        "Even if the relevance scores seem low, carefully read through ALL the provided context "
        "as it has been specifically retrieved for this query. "
        "Extract and synthesize information directly from the context provided. "
        "If you find relevant information in the context, provide a comprehensive answer. "
        "Always mention the source documents you're referencing in your answer. "
        "Only respond with 'I don't know based on the provided context' if the context "
        "truly contains no information related to the question."
    )

    # Debug: Show what context is being sent to the LLM
    # st.write(f"**Debug:** Sending {len(context)} characters to LLM")
    # st.write(f"**Debug:** Context preview: {context[:300]}...")

    try:
        resp = client_oai.chat.completions.create(
            model=CHAT_COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Context:\n{context}\n\nQ: {question}\nA:"}
            ],
            temperature=0.3
        )

        answer = resp.choices[0].message.content.strip()

        # Add source attribution
        if answer != "I don't know based on the provided context.":
            answer += f"\n\n**Sources:** {', '.join(unique_sources)}"

        return answer

    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "I apologize, but I encountered an error while generating the response."


# ========== UI ==========
# User authentication section
username = st.text_input("Enter your username")
if not username:
    st.info("🔑 Please enter your username to get started.")
    st.info(
        "**Available users:** alice (finance), bob (marketing), charlie (hr), dave (engineering), ceo (c_level), eve (employee)")
    st.stop()

role = authenticate(username)
if not role:
    st.error("❌ Invalid user. Try: alice, bob, charlie, dave, ceo, eve")
    st.stop()

st.success(f"✅ Authenticated as **{username}** (Role: {role})")
allowed_cols = collections_for(role)
st.info(f"**Your access:** {', '.join(allowed_cols)}")

# Add debugging toggle
debug_mode = st.checkbox("Enable Debug Mode", value=False)

# Question input area
query = st.text_area("Ask a question:", height=100,
                     placeholder="e.g., What are our marketing expenses? or Tell me about our company policies.")

if st.button("Submit") and query.strip():
    with st.spinner("🔍 Retrieving…"):
        docs = retrieve(allowed_cols, query)

        # If embedding-based retrieval has poor similarity scores, try keyword fallback
        if not docs or all(doc['score'] < 0.4 for doc in docs):
            st.warning("⚠️ **Low embedding similarity detected - trying keyword matching**")
            keyword_docs = keyword_fallback(allowed_cols, query)
            if keyword_docs:
                st.info(f"Found {len(keyword_docs)} documents via keyword matching")
                docs = keyword_docs

        # Apply engineering-specific keyword fallback
        if role == "engineering" and (not docs or all(doc['score'] < 0.4 for doc in docs)):
            if debug_mode:
                st.write("**Debug:** Applying engineering keyword fallback")
            engineering_docs = engineering_keyword_fallback(query)
            if engineering_docs:
                docs = engineering_docs

        # if debug_mode:
        #     st.write(f"**Debug:** Retrieved {len(docs)} documents")
        #     for i, doc in enumerate(docs[:3]):  # Show first 3 for debugging
        #         st.write(f"**Doc {i + 1}:** {doc['content'][:200]}...")

    # Generate answer
    if not docs:
        answer = "I don't know based on the provided context."
        # if debug_mode:
        #     st.warning("**Debug:** No documents retrieved, check if collections have data")
    else:
        answer = rag_answer(query, docs)

    # Display the answer
    st.markdown("### Answer")
    st.markdown("---")

    st.write(
        f"**Domain:** FinTech &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Function:** AI Engineering")
    st.write("")

    st.write(answer)

    # Show document count for debugging
    # if debug_mode and docs:
    #     st.write(f"**Debug:** Answer generated from {len(docs)} documents")

# ========== DIAGNOSTIC SECTION ==========
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Diagnostics")

if st.sidebar.button("Test Database Connection"):
    try:
        collections = client_db.list_collections()
        st.sidebar.success(f"✅ Connected! Found {len(collections)} collections")

        for col in collections:
            count = col.count()
            st.sidebar.write(f"- {col.name}: {count} documents")

    except Exception as e:
        st.sidebar.error(f"❌ Database error: {e}")

if st.sidebar.button("Test OpenAI Connection"):
    try:
        test_embedding = embed(["test"])
        if test_embedding:
            st.sidebar.success("✅ OpenAI API working!")
            st.sidebar.write(f"Embedding dimensions: {len(test_embedding[0])}")
        else:
            st.sidebar.error("❌ OpenAI embedding failed")
    except Exception as e:
        st.sidebar.error(f"❌ OpenAI error: {e}")

# Add sample queries for each role
st.sidebar.markdown("### 💡 Sample Queries")
sample_queries = {
    "finance": ["What are our marketing expenses?", "Show me financial reports", "What are the equipment costs?"],
    "marketing": ["How did our last campaign perform?", "What's the customer feedback?", "Show sales metrics"],
    "hr": ["Show employee attendance", "What are the payroll details?", "Performance review data"],
    "engineering": ["What's our technical architecture?", "Tell me about agile methodology", "Development processes"],
    "c_level": ["Overall company performance", "All department summaries", "Strategic insights"],
    "employee": ["Company policies", "Upcoming events", "General FAQs"]
}

if role in sample_queries:
    st.sidebar.write(f"**For {role}:**")
    for sq in sample_queries[role]:
        if st.sidebar.button(sq, key=f"sample_{sq}"):
            st.experimental_set_query_params(query=sq)

# ========== EMBEDDING CONSISTENCY CHECK ==========
st.sidebar.markdown("### ⚙️ System Status")
st.sidebar.write(f"**Embedding Model:** {EMBEDDING_MODEL}")
st.sidebar.write(f"**Chat Model:** {CHAT_COMPLETION_MODEL}")

if st.sidebar.button("Check Embedding Consistency"):
    try:
        # Test with a simple query
        test_query = "agile methodology"
        test_embedding = embed([test_query])[0]

        # Check embedding dimension
        st.sidebar.write(f"Embedding dimensions: {len(test_embedding)}")
        st.sidebar.write(f"Sample values: {test_embedding[:3]}")

        # Test against engineering collection if available
        if "engineering_docs" in collection_names:
            col = client_db.get_collection("engineering_docs")
            res = col.query(
                query_embeddings=[test_embedding],
                n_results=3,
                include=["distances"]
            )
            distances = res["distances"][0]
            st.sidebar.write(f"Sample distances: {[f'{d:.3f}' for d in distances]}")

            if all(0 <= d <= 2 for d in distances):
                st.sidebar.success("✅ Distance values look normal (0-2 range)")
            else:
                st.sidebar.error("⚠️ Unusual distance values detected!")

    except Exception as e:
        st.sidebar.error(f"Embedding test failed: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Notes")
st.sidebar.markdown("""
**Remember to:**
- Re-run ingestion script after fixes
- Ensure same OpenAI model in both scripts
- Check data folder structure matches departments
""")