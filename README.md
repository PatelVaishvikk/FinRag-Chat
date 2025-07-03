# FinRag-Chat

<!-- README.md -->

# FinTech Insight Architect

**Author:** Vaishvik Patel
**Tech Stack:** Python, FastAPI, Streamlit, ChromaDB, OpenAI API (GPT-4o)

---

## 📖 Project Overview

FinTech Insight Architect is a RAG-based (Retrieval-Augmented Generation) chatbot with Role-Based Access Control (RBAC). It authenticates users, assigns roles, retrieves relevant documents from a vector store, and generates context-rich responses using the OpenAI API. Each user only sees data authorized for their role.

---

## 🛠️ Features

* **Authentication & Role Assignment**

  * Static credential store with username/password and mapped roles.
  * Session-based login flow in Streamlit.

* **Role-Based Access Control (RBAC)**

  * Finance, Marketing, HR, Engineering, C-Level, and Employee roles.
  * Each role can only query its permitted document collections.

* **RAG Pipeline**

  * **Retrieval:** Query ChromaDB vector store using OpenAI embeddings.
  * **Fallback:** Keyword and engineering-specific fallback when similarities are low.
  * **Augmentation:** Assemble top-k document snippets with metadata.
  * **Generation:** Pass context to GPT-4o for final answer, with inline source attribution.

* **Diagnostics & Debugging**

  * Debug mode toggle, database and API connection tests, sample queries per role.

---

## 🏗️ Architecture

```text
+---------------+      +-----------------------+      +-------------------+
|  Streamlit UI |<---->|   FastAPI Backend     |<---->| ChromaDB Vector   |
| (frontend)    |      | (optional middleware) |      | Store (docs)      |
+---------------+      +-----------------------+      +-------------------+
        |                       |                              |
        v                       v                              v
  Username/password      Validate credentials            Embeddings & query
  → role in session      → JWT/session cookie           → retrieve top-k docs
        |                       |                              |
        +------------------------------------------------------+    
                               RAG Logic
        +------------------------------------------------------+    
        |                       |                              |
        v                       v                              v
  OpenAI Embeddings → Retrieve docs → Assemble context → GPT-4o chat
                                                    → Answer + citations
```

---

## 🔎 How It Works

1. **Login (Streamlit):** Users enter username & password in the Streamlit UI.
2. **Authenticate (app.auth):** The `authenticate()` function in `app/auth.py` verifies credentials against the static `_CREDENTIALS` store.
3. **Role Assignment:** On success, Streamlit stores the user's role in `st.session_state.role`.
4. **Query Submission:** Users type a natural language query into the chat interface.
5. **Retrieval (FastAPI / inline):**

   * **Embedding:** The query is embedded via the OpenAI API.
   * **Vector Search:** ChromaDB is queried for the top‑k relevant document chunks, filtered by the user’s allowed collections based on their role.
6. **Augmentation:** Retrieved document snippets are assembled into a context window, each with source metadata.
7. **Generation (OpenAI GPT-4o):** The assembled context and user question are sent to GPT-4o to generate a concise, context‑rich answer.
8. **Display:** Streamlit renders the final answer along with inline source citations.

---

## 🚀 Setup & Installation

1. **Clone the Repo**

   ```bash
   git clone https://github.com/<your-username>/fintech-insight-architect.git
   cd fintech-insight-architect
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure OpenAI API Key**

   * Create a `.env` file in `frontend/` or set `OPENAI_API_KEY` env var.

   ```ini
   OPENAI_API_KEY=sk-...
   ```

5. **Run Ingestion**

   ```bash
   python ingest_docs.py
   ```

   * Splits data files into chunks, generates embeddings, and populates ChromaDB.

6. **Launch Streamlit App**

   ```bash
   streamlit run frontend/streamlit_appp.py
   ```

---

## 🔧 Usage

1. Navigate to `http://localhost:8501`.

2. **Log in** with one of the provided credentials:

   * `alice` / `alice123` → Finance
   * `bob` / `bob123` → Marketing
   * `charlie` / `charlie123` → HR
   * `dave` / `dave123` → Engineering
   * `ceo` / `c3oP@ss` → C-Level
   * `eve` / `eve123` → Employee

3. Once logged in, enter a natural language query.

4. The chatbot retrieves relevant docs and generates an answer with sources.

---

## 📝 Evaluation Criteria

* **Accuracy:** Correct role enforcement and RAG responses.
* **Security:** Credentials handling and RBAC gating.
* **User Experience:** Clear login flow, diagnostic tools, and sample queries.
* **Presentation:** README clarity, video demo, and LinkedIn post.

---

## 🎥 Demo & Presentation

* **Video (≤15 min):** Walkthrough architecture, demo queries per role, diagnostics.
* **LinkedIn Post:** Link to GitHub, video, and personal reflection.

---

## 🤝 Contributing

Feel free to submit issues or pull requests. For major changes, open an issue first to discuss.

---

© 2025 Vaishvik Patel
