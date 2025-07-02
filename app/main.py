import os
import jwt
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import hashlib
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="FinRAG Chatbot API", version="1.0.0")

# Add CORS middleware for Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# =============================================================================
# CONFIGURATION
# =============================================================================
JWT_SECRET = "yLFN2nFRsqjxL77X0ejxrRTQTpb_BN7gYYl0DN4fEN4"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# OpenAI Setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

client_oai = OpenAI(api_key=OPENAI_API_KEY)
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"

# ChromaDB Setup
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../chroma_db"))
chroma_client = chromadb.Client(Settings(
    persist_directory=DB_PATH,
    is_persistent=True,
    anonymized_telemetry=False
))


# =============================================================================
# MODELS & ENUMS
# =============================================================================
class UserRole(str, Enum):
    FINANCE = "finance"
    MARKETING = "marketing"
    HR = "hr"
    ENGINEERING = "engineering"
    C_LEVEL = "c_level"
    EMPLOYEE = "employee"


class LoginRequest(BaseModel):
    username: str
    password: str


class QueryRequest(BaseModel):
    question: str
    max_results: Optional[int] = 10  # Allow more results for better context


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user_role: str
    username: str
    accessible_departments: List[str]


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[str]
    user_role: str
    confidence_score: float
    departments_searched: List[str]
    debug_info: Optional[Dict[str, Any]] = None  # For debugging


class User(BaseModel):
    username: str
    role: UserRole
    department: str
    hashed_password: str


# =============================================================================
# USER DATABASE & RBAC
# =============================================================================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# Mock user database
USERS_DB = {
    "alice": User(username="alice", role=UserRole.FINANCE, department="Finance",
                  hashed_password=hash_password("fin123")),
    "bob": User(username="bob", role=UserRole.MARKETING, department="Marketing",
                hashed_password=hash_password("mkt123")),
    "charlie": User(username="charlie", role=UserRole.HR, department="HR",
                    hashed_password=hash_password("hr123")),
    "dave": User(username="dave", role=UserRole.ENGINEERING, department="Engineering",
                 hashed_password=hash_password("eng123")),
    "ceo": User(username="ceo", role=UserRole.C_LEVEL, department="Executive",
                hashed_password=hash_password("ceo123")),
    "eve": User(username="eve", role=UserRole.EMPLOYEE, department="General",
                hashed_password=hash_password("emp123"))
}

# Role-based access control mapping
ROLE_COLLECTIONS = {
    UserRole.FINANCE: ["finance_docs", "general_docs"],
    UserRole.MARKETING: ["marketing_docs", "general_docs"],
    UserRole.HR: ["hr_docs", "general_docs"],
    UserRole.ENGINEERING: ["engineering_docs", "general_docs"],
    UserRole.C_LEVEL: ["finance_docs", "marketing_docs", "hr_docs", "engineering_docs", "general_docs"],
    UserRole.EMPLOYEE: ["general_docs"]
}


class RBACManager:
    @staticmethod
    def get_accessible_collections(user_role: UserRole) -> List[str]:
        return ROLE_COLLECTIONS.get(user_role, [])

    @staticmethod
    def can_access_data(user_role: UserRole, collection: str) -> bool:
        return collection in RBACManager.get_accessible_collections(user_role)


# =============================================================================
# AUTHENTICATION FUNCTIONS
# =============================================================================
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user info"""
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")

        if username is None or role is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        user = USERS_DB.get(username)
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")

        return {"username": username, "role": UserRole(role), "user": user}

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# =============================================================================
# IMPROVED RAG ENGINE
# =============================================================================
class RAGEngine:
    def __init__(self):
        self.rbac_manager = RBACManager()
        # Lower confidence threshold for better recall
        self.min_confidence_threshold = 0.15  # Reduced from 0.3

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text"""
        try:
            response = client_oai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

    def retrieve_documents(self, query: str, allowed_collections: List[str], top_k: int = 10) -> tuple[
        List[Dict], Dict]:
        """Retrieve relevant documents from allowed collections with better error handling"""
        if not allowed_collections:
            return [], {"error": "No allowed collections"}

        # Generate query embedding
        try:
            query_embedding = self.embed_text([query])[0]
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return [], {"error": f"Query embedding failed: {e}"}

        all_results = []
        debug_info = {"collections_tried": [], "collections_found": [], "total_docs": 0}

        for collection_name in allowed_collections:
            debug_info["collections_tried"].append(collection_name)
            try:
                # Check if collection exists
                try:
                    collection = chroma_client.get_collection(collection_name)
                    collection_count = collection.count()
                    debug_info["collections_found"].append(collection_name)
                    debug_info["total_docs"] += collection_count

                    logger.info(f"Collection {collection_name} has {collection_count} documents")

                    if collection_count == 0:
                        logger.warning(f"Collection {collection_name} is empty")
                        continue

                except Exception as e:
                    logger.warning(f"Collection {collection_name} not found or inaccessible: {e}")
                    continue

                # Query the collection with better parameters
                try:
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=min(top_k, collection_count),
                        include=["documents", "distances", "metadatas"]
                    )

                    if not results["documents"] or not results["documents"][0]:
                        logger.warning(f"No results returned from collection {collection_name}")
                        continue

                    # Process results
                    for i, (doc, distance) in enumerate(zip(results["documents"][0], results["distances"][0])):
                        # Better similarity calculation
                        # ChromaDB uses cosine distance, convert to similarity
                        similarity_score = max(0, 1 - distance)

                        metadata = results.get("metadatas", [[{}]])[0][i] if results.get("metadatas") else {}

                        all_results.append({
                            "content": doc.strip(),
                            "similarity_score": similarity_score,
                            "source": collection_name,
                            "metadata": metadata,
                            "distance": distance
                        })

                        logger.debug(
                            f"Document from {collection_name}: similarity={similarity_score:.3f}, distance={distance:.3f}")

                except Exception as e:
                    logger.error(f"Error querying collection {collection_name}: {e}")
                    continue

            except Exception as e:
                logger.error(f"Error accessing collection {collection_name}: {e}")
                continue

        # Sort by similarity score and return top results
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)

        debug_info["total_results"] = len(all_results)
        debug_info["top_similarities"] = [r["similarity_score"] for r in all_results[:5]]

        logger.info(
            f"Retrieved {len(all_results)} total results, top similarity: {all_results[0]['similarity_score']:.3f}" if all_results else "No results retrieved")

        return all_results[:top_k], debug_info

    def generate_answer(self, query: str, documents: List[Dict], user_role: str) -> Dict[str, Any]:
        """Generate answer using retrieved documents with improved logic"""

        # Filter documents by confidence threshold
        relevant_docs = [doc for doc in documents if doc["similarity_score"] >= self.min_confidence_threshold]

        logger.info(
            f"Documents above threshold ({self.min_confidence_threshold}): {len(relevant_docs)} out of {len(documents)}")

        # If no documents meet the threshold, try with lower threshold or use all available
        if not relevant_docs and documents:
            # Use all documents but with a warning
            relevant_docs = documents[:5]  # Use top 5 regardless of score
            logger.warning(
                f"Using documents below confidence threshold. Highest score: {documents[0]['similarity_score']:.3f}")

        if not relevant_docs:
            return {
                "answer": f"I don't have access to relevant information in your authorized document collections ({user_role} role) to answer this question. Please check if:\n1. The question relates to your department's domain\n2. The relevant documents have been uploaded to the system\n3. You have the necessary permissions to access this information",
                "confidence_score": 0.0,
                "sources": [],
                "access_denied": True,
                "reason": "No relevant documents found"
            }

        # Prepare context from documents
        context_parts = []
        sources = set()

        # Use more documents for better context
        num_docs_to_use = min(5, len(relevant_docs))

        for i, doc in enumerate(relevant_docs[:num_docs_to_use]):
            content = doc["content"]
            source = doc["source"]
            score = doc["similarity_score"]

            # Clean and truncate content if too long
            content = content.strip()
            if len(content) > 1000:  # Truncate very long documents
                content = content[:1000] + "..."

            context_parts.append(f"Document {i + 1} (Source: {source}, Relevance: {score:.2f}):\n{content}")
            sources.add(source)

        context = "\n\n" + "=" * 50 + "\n\n".join(context_parts)

        # Calculate overall confidence
        confidence_score = sum(doc["similarity_score"] for doc in relevant_docs[:num_docs_to_use]) / num_docs_to_use

        # Improved system prompt
        system_prompt = f"""You are a helpful assistant for FinSolve Technologies with role-based access control.

IMPORTANT INSTRUCTIONS:
1. Answer the user's question using ONLY the provided context documents
2. The user has '{user_role}' role - provide information relevant to their role
3. Be comprehensive and detailed when the context supports it
4. If the context has relevant information but it's partial, provide what you can and mention limitations
5. Do NOT refuse to answer if there is relevant information in the context
6. Format your response clearly and professionally
7. If you need to make connections between different pieces of information in the context, that's allowed

Context Quality: The documents provided have been pre-filtered for relevance to the user's question."""

        try:
            response = client_oai.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context Documents:\n{context}\n\n" +
                                                f"Question: {query}\n\n" +
                                                f"Please provide a comprehensive answer based on the context above:"}
                ],
                temperature=0.2,  # Slightly higher for more natural responses
                max_tokens=800  # Increased for more detailed answers
            )

            answer = response.choices[0].message.content.strip()

            # Quality check: ensure the answer is substantial
            if len(answer) < 50:
                logger.warning(f"Generated answer is very short: {len(answer)} characters")

            return {
                "answer": answer,
                "confidence_score": confidence_score,
                "sources": list(sources),
                "access_denied": False,
                "documents_used": num_docs_to_use,
                "total_relevant_docs": len(relevant_docs)
            }

        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")


# Initialize RAG engine
rag_engine = RAGEngine()


# =============================================================================
# API ENDPOINTS
# =============================================================================
@app.post("/auth/login", response_model=LoginResponse)
async def login(login_data: LoginRequest):
    """Authenticate user and return JWT token"""
    username = login_data.username.lower().strip()

    # Check if user exists
    user = USERS_DB.get(username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Verify password
    if user.hashed_password != hash_password(login_data.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Create access token
    access_token_expires = timedelta(hours=JWT_EXPIRATION_HOURS)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role.value},
        expires_delta=access_token_expires
    )

    # Get accessible departments
    accessible_departments = RBACManager.get_accessible_collections(user.role)

    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        user_role=user.role.value,
        username=user.username,
        accessible_departments=accessible_departments
    )


@app.post("/chat/query", response_model=QueryResponse)
async def process_query(query_data: QueryRequest, current_user: dict = Depends(verify_token)):
    """Process user query and return RAG-based response with improved retrieval"""
    user = current_user["user"]
    user_role = current_user["role"]

    logger.info(f"Processing query from {user.username} ({user_role.value}): '{query_data.question[:100]}...'")

    # Get accessible collections for user role
    accessible_collections = RBACManager.get_accessible_collections(user_role)

    if not accessible_collections:
        raise HTTPException(status_code=403, detail="No data access permissions for your role")

    # Retrieve relevant documents with debug info
    documents, debug_info = rag_engine.retrieve_documents(
        query=query_data.question,
        allowed_collections=accessible_collections,
        top_k=query_data.max_results
    )

    # Generate answer
    result = rag_engine.generate_answer(
        query=query_data.question,
        documents=documents,
        user_role=user_role.value
    )

    # Enhanced logging
    logger.info(f"Query result - User: {user.username}, "
                f"Documents found: {len(documents)}, "
                f"Confidence: {result['confidence_score']:.3f}, "
                f"Access denied: {result.get('access_denied', False)}")

    return QueryResponse(
        query=query_data.question,
        answer=result["answer"],
        sources=result["sources"],
        user_role=user_role.value,
        confidence_score=result["confidence_score"],
        departments_searched=accessible_collections,
        debug_info={
            "retrieval_debug": debug_info,
            "documents_processed": len(documents),
            "documents_used": result.get("documents_used", 0),
            "min_threshold": rag_engine.min_confidence_threshold
        }
    )


@app.get("/auth/verify")
async def verify_auth(current_user: dict = Depends(verify_token)):
    """Verify if current token is valid"""
    return {
        "valid": True,
        "username": current_user["username"],
        "role": current_user["role"].value,
        "accessible_collections": RBACManager.get_accessible_collections(current_user["role"])
    }


@app.get("/debug/collections")
async def debug_collections(current_user: dict = Depends(verify_token)):
    """Debug endpoint to check collection status"""
    try:
        user_role = current_user["role"]
        accessible_collections = RBACManager.get_accessible_collections(user_role)

        collection_info = {}
        for collection_name in accessible_collections:
            try:
                collection = chroma_client.get_collection(collection_name)
                count = collection.count()
                collection_info[collection_name] = {
                    "exists": True,
                    "document_count": count,
                    "status": "accessible"
                }
            except Exception as e:
                collection_info[collection_name] = {
                    "exists": False,
                    "error": str(e),
                    "status": "error"
                }

        return {
            "user_role": user_role.value,
            "accessible_collections": accessible_collections,
            "collection_details": collection_info,
            "total_available_docs": sum(info.get("document_count", 0) for info in collection_info.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug check failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test ChromaDB connection
        collections = chroma_client.list_collections()

        # Test OpenAI connection
        test_embedding = client_oai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=["test"]
        )

        return {
            "status": "healthy",
            "chroma_collections": len(collections),
            "available_collections": [c.name for c in collections],
            "openai_embedding_dim": len(test_embedding.data[0].embedding),
            "min_confidence_threshold": rag_engine.min_confidence_threshold
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")