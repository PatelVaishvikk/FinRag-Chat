import streamlit as st
import requests
import json
from typing import Optional
from datetime import datetime
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="FinRAG Chatbot",
    page_icon="",
    layout="wide"
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap" rel="stylesheet">

<style>
  :root {
    --primary: #1f4e79;
    --secondary: #28a745;
    --warning: #ffc107;
    --error: #dc3545;
    --light-bg: #f8f9fa;
    --alt-bg: #f0f7ff;
    --radius: 0.75rem;
    --gap: 1.5rem;
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    font-family: 'Inter', sans-serif;
  }

  body {
    background: #eef4fb;
    color: #333;
    line-height: 1.6;
    padding: 2rem;
  }

  .main-header {
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: white;
    padding: var(--gap) 1rem;
    border-radius: var(--radius);
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    margin-bottom: calc(var(--gap) * 1.5);
    transition: transform 0.3s ease;
  }
  .main-header:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 24px rgba(0,0,0,0.15);
  }

  .user-info, .warning-box, .error-box {
    padding: var(--gap);
    border-radius: var(--radius);
    margin-bottom: var(--gap);
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    transition: box-shadow 0.3s ease;
  }
  .user-info {
    background: var(--alt-bg);
    border-left: 6px solid var(--primary);
  }
  .user-info:hover {
    box-shadow: 0 6px 16px rgba(0,0,0,0.08);
  }

  .query-box {
    margin: var(--gap) 0;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  .query-box input,
  .query-box textarea {
    padding: 0.75rem 1rem;
    border: 1px solid #ccc;
    border-radius: var(--radius);
    outline: none;
    font-size: 1rem;
    transition: border-color 0.2s;
  }
  .query-box input:focus,
  .query-box textarea:focus {
    border-color: var(--primary);
    box-shadow: 0 0 8px rgba(31,78,121,0.2);
  }

  .answer-box {
    background: var(--light-bg);
    border-left: 6px solid var(--secondary);
    padding: var(--gap);
    border-radius: var(--radius);
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    transition: box-shadow 0.3s;
  }
  .answer-box:hover {
    box-shadow: 0 6px 16px rgba(0,0,0,0.08);
  }

  .warning-box {
    background: #fffceb;
    border-left: 6px solid var(--warning);
  }

  .error-box {
    background: #fbeaea;
    border-left: 6px solid var(--error);
  }

  .debug-info {
    background: #e7f3ff;
    padding: 0.75rem;
    border-radius: 0.5rem;
    font-size: 0.85rem;
    color: #555;
    margin-top: 1rem;
    overflow-x: auto;
  }

  @media (max-width: 600px) {
    .main-header { font-size: 2rem; padding: 1rem; }
    body { padding: 1rem; }
  }
</style>

""", unsafe_allow_html=True)

# Initialize session state
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_debug' not in st.session_state:
    st.session_state.show_debug = False


def make_api_request(endpoint: str, method: str = "GET", data: dict = None, auth_required: bool = False):
    url = f"{API_BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}

    if auth_required and st.session_state.access_token:
        headers["Authorization"] = f"Bearer {st.session_state.access_token}"

    try:
        if method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=30)
        else:
            response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            return response.json(), None
        elif response.status_code == 401:

            st.session_state.access_token = None
            st.session_state.user_info = None
            return None, "Session expired. Please login again."
        else:
            try:
                error_detail = response.json().get("detail", f"HTTP {response.status_code}")
            except:
                error_detail = f"HTTP {response.status_code}"
            return None, error_detail
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend server. Please ensure FastAPI server is running on http://localhost:8000"
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except Exception as e:
        return None, str(e)


def login(username: str, password: str):

    data, error = make_api_request(
        "/auth/login",
        method="POST",
        data={"username": username, "password": password}
    )

    if data:
        st.session_state.access_token = data["access_token"]
        st.session_state.user_info = data
        return True, "Login successful!"
    else:
        return False, error


def logout():
    """Logout user"""
    st.session_state.access_token = None
    st.session_state.user_info = None
    st.session_state.chat_history = []


def send_query(question: str, max_results: int = 10):

    data, error = make_api_request(
        "/chat/query",
        method="POST",
        data={"question": question, "max_results": max_results},
        auth_required=True
    )

    if data:
        return data, None
    else:
        return None, error


def check_system_health():

    data, error = make_api_request("/health", auth_required=True)
    return data, error


def get_debug_info():

    data, error = make_api_request("/debug/collections", auth_required=True)
    return data, error


def verify_token():

    data, error = make_api_request("/auth/verify", auth_required=True)
    return data, error



st.markdown('<h1 class="main-header">🏦 FinRAG Chatbot</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: #666; font-size: 1.2rem;">Secure Role-Based Access to Company Data</p>',
    unsafe_allow_html=True)


if st.session_state.access_token:

    token_data, token_error = verify_token()
    if token_error:
        st.error(f"Session invalid: {token_error}")
        logout()
        st.rerun()


if not st.session_state.access_token:
    st.markdown("### 🔐 Authentication Required")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("Login", use_container_width=True)

            if submit_button:
                if username and password:
                    with st.spinner("Authenticating..."):
                        success, message = login(username, password)

                    if success:
                        st.success(message)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Login failed: {message}")
                else:
                    st.error("Please enter both username and password")


    with st.expander("🔧 System Status"):
        if st.button("Check Backend Status"):
            health_data, health_error = make_api_request("/health")
            if health_data:
                st.success("✅ Backend is healthy")
                st.json({
                    "Collections Available": health_data.get("chroma_collections", 0),
                    "OpenAI Embedding Dimension": health_data.get("openai_embedding_dim", "N/A"),
                    "Available Collections": health_data.get("available_collections", [])
                })
            else:
                st.error(f"❌ Backend health check failed: {health_error}")


    st.markdown("---")
    # st.markdown("### 👥 Sample User Credentials")
    #
    # credentials_col1, credentials_col2 = st.columns(2)
    #
    # with credentials_col1:
    #     st.markdown("""
    #     **Finance Team:**
    #     - Username: `alice` | Password: `fin123`
    #
    #     **Marketing Team:**
    #     - Username: `bob` | Password: `mkt123`
    #
    #     **HR Team:**
    #     - Username: `charlie` | Password: `hr123`
    #     """)
    #
    # with credentials_col2:
    #     st.markdown("""
    #     **Engineering Team:**
    #     - Username: `dave` | Password: `eng123`
    #
    #     **C-Level Executive:**
    #     - Username: `ceo` | Password: `ceo123`
    #
    #     **General Employee:**
    #     - Username: `eve` | Password: `emp123`
    #     """)

else:

    user_info = st.session_state.user_info


    st.markdown('<div class="user-info">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

    with col1:
        st.markdown(f"**👤 Welcome, {user_info['username'].title()}!**")
        st.markdown(f"**🎯 Role:** {user_info['user_role'].title()}")

    with col2:
        st.markdown("**📁 Access to:**")
        for dept in user_info['accessible_departments']:
            st.markdown(f"• {dept.replace('_docs', '').title()}")

    with col3:
        if st.button("🔧 Debug", use_container_width=True):
            st.session_state.show_debug = not st.session_state.show_debug

    with col4:
        if st.button("🚪 Logout", use_container_width=True):
            logout()
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.show_debug:
        st.markdown("### 🔍 Debug Information")
        debug_data, debug_error = get_debug_info()

        if debug_data:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Collection Status:**")
                for coll_name, coll_info in debug_data['collection_details'].items():
                    if coll_info['exists']:
                        st.success(f"✅ {coll_name}: {coll_info['document_count']} docs")
                    else:
                        st.error(f"❌ {coll_name}: {coll_info.get('error', 'Not found')}")

            with col2:
                st.metric("Total Available Documents", debug_data.get('total_available_docs', 0))
                st.markdown(f"**Your Role:** {debug_data['user_role']}")
                st.markdown(f"**Accessible Collections:** {len(debug_data['accessible_collections'])}")
        else:
            st.error(f"Debug info failed: {debug_error}")


    st.markdown('<div class="query-box">', unsafe_allow_html=True)
    st.markdown("### 💬 Ask a Question")


    with st.expander("⚙️ Advanced Options"):
        max_results = st.slider("Maximum results to retrieve", 1, 20, 10,
                                help="More results may provide better context but take longer")
        show_debug_output = st.checkbox("Show debug information in responses")


    role_questions = {
        "finance": [
            "What are our quarterly financial results?",
            "Show me budget allocation details",
            "What are the major expense categories?",
            "Tell me about revenue trends"
        ],
        "marketing": [
            "How did our recent campaigns perform?",
            "What's the customer acquisition cost?",
            "Show me market analysis data",
            "Tell me about brand performance"
        ],
        "hr": [
            "What's our employee turnover rate?",
            "Show me recruitment metrics",
            "Tell me about employee satisfaction",
            "What are the training completion rates?"
        ],
        "engineering": [
            "What's our current technical stack?",
            "Tell me about development methodologies",
            "Show me system architecture details",
            "What are our deployment processes?"
        ],
        "c_level": [
            "Give me a company-wide performance overview",
            "What are the key strategic metrics?",
            "Show me cross-departmental insights",
            "Tell me about overall business health"
        ],
        "employee": [
            "What are the company policies?",
            "Tell me about employee benefits",
            "Show me general company information",
            "What training opportunities are available?"
        ]
    }


    if user_info['user_role'] in role_questions:
        st.markdown("**💡 Sample Questions for your role:**")
        questions = role_questions[user_info['user_role']]

        q_col1, q_col2 = st.columns(2)
        for i, question in enumerate(questions):
            col = q_col1 if i % 2 == 0 else q_col2
            with col:
                if st.button(question, key=f"sample_{i}", use_container_width=True):
                    st.session_state.current_query = question

    query = st.text_area(
        "Your Question:",
        height=100,
        placeholder="Type your question here... Be specific for better results!",
        value=st.session_state.get('current_query', '')
    )

    if 'current_query' in st.session_state:
        del st.session_state.current_query

    if st.button("🔍 Submit Query", use_container_width=True, type="primary"):
        if query.strip():
            with st.spinner("Processing your query..."):
                start_time = time.time()
                response, error = send_query(query, max_results)
                end_time = time.time()

            if response:

                st.session_state.chat_history.append({
                    "query": query,
                    "response": response,
                    "timestamp": str(datetime.now()),
                    "processing_time": end_time - start_time
                })


                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.markdown("### 📝 Answer")
                st.markdown(response['answer'])


                if response['confidence_score'] < 0.3:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.warning(
                        "⚠️ **Low Confidence Response**: The system found limited relevant information. Consider rephrasing your question or checking if the topic is covered in your accessible documents.")
                    st.markdown('</div>', unsafe_allow_html=True)


                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    confidence_color = "green" if response['confidence_score'] > 0.7 else "orange" if response[
                                                                                                          'confidence_score'] > 0.3 else "red"
                    st.metric("Confidence Score", f"{response['confidence_score']:.2f}")

                with col2:
                    st.metric("Processing Time", f"{end_time - start_time:.1f}s")

                with col3:
                    st.markdown("**Sources Used:**")
                    if response['sources']:
                        for source in response['sources']:
                            st.markdown(f"• {source.replace('_docs', '').title()}")
                    else:
                        st.markdown("• None found")

                with col4:
                    st.markdown("**Departments Searched:**")
                    for dept in response['departments_searched']:
                        st.markdown(f"• {dept.replace('_docs', '').title()}")


                if show_debug_output and response.get('debug_info'):
                    st.markdown('<div class="debug-info">', unsafe_allow_html=True)
                    st.markdown("**🐛 Debug Information:**")
                    debug_info = response['debug_info']

                    if 'retrieval_debug' in debug_info:
                        ret_debug = debug_info['retrieval_debug']
                        st.write(f"Collections tried: {ret_debug.get('collections_tried', [])}")
                        st.write(f"Collections found: {ret_debug.get('collections_found', [])}")
                        st.write(f"Total documents: {ret_debug.get('total_docs', 0)}")
                        st.write(f"Results returned: {ret_debug.get('total_results', 0)}")
                        if ret_debug.get('top_similarities'):
                            st.write(f"Top similarities: {[f'{s:.3f}' for s in ret_debug['top_similarities']]}")

                    st.write(f"Documents processed: {debug_info.get('documents_processed', 0)}")
                    st.write(f"Documents used: {debug_info.get('documents_used', 0)}")
                    st.write(f"Min threshold: {debug_info.get('min_threshold', 0)}")
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.markdown('<div class="error-box">', unsafe_allow_html=True)
                st.error(f"❌ Query failed: {error}")


                if "No relevant documents" in str(error):
                    st.markdown("**💡 Suggestions:**")
                    st.markdown("• Try rephrasing your question")
                    st.markdown("• Use more general terms")
                    st.markdown("• Check if the topic is within your role's access")
                elif "Session expired" in str(error):
                    st.markdown("**🔄 Please login again to continue**")
                    if st.button("Refresh Page"):
                        st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter a question")

    st.markdown('</div>', unsafe_allow_html=True)


    if st.session_state.chat_history:
        st.markdown("### 📚 Recent Chat History")


        col1, col2 = st.columns([3, 1])
        with col1:
            history_limit = st.selectbox("Show last", [5, 10, 20], index=0)
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()


        for i, chat in enumerate(reversed(st.session_state.chat_history[-history_limit:])):
            timestamp = datetime.fromisoformat(chat['timestamp']).strftime("%H:%M:%S")
            confidence = chat['response']['confidence_score']
            confidence_emoji = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.3 else "🔴"

            with st.expander(f"{confidence_emoji} [{timestamp}] {chat['query'][:60]}..."):
                st.markdown(f"**Query:** {chat['query']}")
                st.markdown(f"**Answer:** {chat['response']['answer']}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", f"{confidence:.2f}")
                with col2:
                    st.metric("Processing Time", f"{chat.get('processing_time', 0):.1f}s")
                with col3:
                    if chat['response']['sources']:
                        st.write(
                            "**Sources:** " + ", ".join([s.replace('_docs', '') for s in chat['response']['sources']]))


st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
with footer_col2:
    st.markdown(
        '<p style="text-align: center; color: #666;">🏦 FinSolve Technologies - AI-Powered Financial Insights</p>',
        unsafe_allow_html=True)


    if st.session_state.access_token and user_info.get('user_role') == 'c_level':
        if st.button("📊 Quick System Health Check", key="footer_health"):
            health_data, health_error = check_system_health()
            if health_data:
                st.success(f"✅ System healthy - {health_data.get('chroma_collections', 0)} collections available")
            else:
                st.error(f"❌ System issue: {health_error}")