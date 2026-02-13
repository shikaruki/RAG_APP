"""
RAG Chatbot with Multi-Document Support
Runs FastAPI backend internally for single-service deployment.
"""

import streamlit as st
import requests
import time
import os
import threading
import uvicorn

# Start FastAPI backend in background thread (only once)
def start_backend():
    from app.main import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

if "backend_started" not in st.session_state:
    st.session_state.backend_started = True
    thread = threading.Thread(target=start_backend, daemon=True)
    thread.start()
    time.sleep(2)  # Wait for backend to start

# API URL points to internal backend
API_URL = "http://127.0.0.1:8000"

# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None
if "selected_doc_name" not in st.session_state:
    st.session_state.selected_doc_name = "All Documents"

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
    }
    
    .bot-message {
        background: #f0f2f6;
        color: #1f2937;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        max-width: 80%;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        text-align: center;
        color: white;
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    /* Upload area */
    .upload-section {
        background: #f8fafc;
        padding: 30px;
        border-radius: 15px;
        border: 2px dashed #cbd5e1;
        text-align: center;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 30px;
        border-radius: 25px;
        font-weight: 600;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

def get_status():
    """Get the status of the knowledge base."""
    try:
        response = requests.get(f"{API_URL}/ingestion/status", timeout=5)
        return response.json()
    except:
        return {"documents": 0, "document_list": [], "status": "offline"}


def get_documents():
    """Get list of all documents."""
    try:
        response = requests.get(f"{API_URL}/ingestion/documents", timeout=5)
        return response.json().get("documents", [])
    except:
        return []


def upload_document(file):
    """Upload a document and get job ID (returns immediately)."""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_URL}/ingestion/upload", files=files, timeout=30)
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_job_status(job_id):
    """Poll job status."""
    try:
        response = requests.get(f"{API_URL}/ingestion/job/{job_id}", timeout=5)
        return response.json()
    except:
        return {"status": "error"}


def ask_question(question, doc_id=None, history=None):
    """Ask a question about a specific document or all documents."""
    try:
        # Format history for API
        formatted_history = []
        if history:
            for msg in history:
                formatted_history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        response = requests.post(
            f"{API_URL}/chat/ask",
            json={"question": question, "doc_id": doc_id, "history": formatted_history},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"answer": f"Connection error: {str(e)}", "sources": []}
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "sources": []}


def delete_doc(doc_id):
    """Delete a specific document."""
    try:
        response = requests.delete(f"{API_URL}/ingestion/documents/{doc_id}", timeout=10)
        return response.json()
    except:
        return {"status": "error"}


def clear_documents():
    """Clear all documents."""
    try:
        response = requests.delete(f"{API_URL}/ingestion/clear", timeout=10)
        return response.json()
    except:
        return {"status": "error"}


# Sidebar
with st.sidebar:
    st.markdown("## 📚 Documents")
    
    # Get documents list
    documents = get_documents()
    
    # Status indicator
    if not documents:
        st.info("📭 No documents uploaded yet")
    else:
        st.success(f"✅ {len(documents)} document(s) available")
    
    st.markdown("---")
    
    # Document selector
    if documents:
        st.markdown("### 📄 Select Document to Chat")
        
        # Build options
        doc_options = {"all": "📚 All Documents"}
        for doc in documents:
            doc_options[doc["id"]] = f"📄 {doc['name'][:25]}... ({doc['chunks']} chunks)" if len(doc['name']) > 25 else f"📄 {doc['name']} ({doc['chunks']} chunks)"
        
        selected = st.selectbox(
            "Choose document:",
            options=list(doc_options.keys()),
            format_func=lambda x: doc_options[x],
            key="doc_selector"
        )
        
        # Update session state
        if selected == "all":
            st.session_state.selected_doc = None
            st.session_state.selected_doc_name = "All Documents"
        else:
            st.session_state.selected_doc = selected
            for doc in documents:
                if doc["id"] == selected:
                    st.session_state.selected_doc_name = doc["name"]
                    break
        
        # Show current selection
        st.caption(f"💬 Chatting with: **{st.session_state.selected_doc_name}**")
        
        # Delete selected document button
        if st.session_state.selected_doc:
            if st.button("🗑️ Delete This Document", use_container_width=True):
                delete_doc(st.session_state.selected_doc)
                st.session_state.selected_doc = None
                st.session_state.selected_doc_name = "All Documents"
                st.session_state.messages = []
                st.rerun()
        
        st.markdown("---")
    
    # Upload section
    st.markdown("### 📤 Upload New Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF document (max 30MB)"
    )
    
    if uploaded_file:
        if st.button("🚀 Process Document", use_container_width=True):
            # Start upload and get job ID
            result = upload_document(uploaded_file)
            
            if result.get("status") == "processing":
                job_id = result.get("job_id")
                st.session_state.current_job_id = job_id
                st.session_state.current_job_filename = result.get("filename")
                st.rerun()
            elif result.get("status") == "error":
                st.error(f"❌ {result.get('message', 'Upload failed')}")
    
    # Show job progress if there's an active job
    if "current_job_id" in st.session_state and st.session_state.current_job_id:
        job_id = st.session_state.current_job_id
        job = get_job_status(job_id)
        
        if job.get("status") == "processing" or job.get("status") == "pending":
            st.info(f"⏳ Processing: {st.session_state.current_job_filename}")
            progress = job.get("progress", 0)
            st.progress(progress / 100, text=job.get("message", "Processing..."))
            time.sleep(2)  # Poll every 2 seconds
            st.rerun()
        elif job.get("status") == "completed":
            st.success(f"✅ Ingestion complete!")
            st.caption(f"Doc ID: {job.get('doc_id')} | {job.get('chunks')} chunks | {job.get('pages')} pages")
            # Auto-select the new document
            st.session_state.selected_doc = job.get('doc_id')
            st.session_state.selected_doc_name = job.get('doc_name')
            st.session_state.messages = []
            st.session_state.current_job_id = None
            time.sleep(1)
            st.rerun()
        elif job.get("status") == "failed":
            st.error(f"❌ Failed: {job.get('error', 'Unknown error')}")
            st.session_state.current_job_id = None
    
    st.markdown("---")
    
    # Clear all button
    if documents:
        if st.button("🗑️ Clear All Documents", use_container_width=True, type="secondary"):
            with st.spinner("Clearing..."):
                clear_documents()
                st.session_state.messages = []
                st.session_state.selected_doc = None
                st.session_state.selected_doc_name = "All Documents"
                st.rerun()
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Multi-Doc RAG Chatbot**
    - � Upload multiple PDFs
    - 🔄 Switch between documents
    - � Chat with context
    """)


# Main content
st.markdown(f"""
<div class="header-container">
    <h1>🤖 RAG Chatbot</h1>
    <p>Chatting with: <strong>{st.session_state.selected_doc_name}</strong></p>
</div>
""", unsafe_allow_html=True)

# Chat interface
chat_container = st.container()

with chat_container:
    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(msg["sources"], 1):
                            st.markdown(f"**Source {i}:** {source}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Get response with chat history and document filter
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            response = ask_question(
                prompt, 
                doc_id=st.session_state.selected_doc,
                history=st.session_state.messages
            )
            answer = response.get("answer", "Sorry, I couldn't get a response.")
            sources = response.get("sources", [])
            
            st.markdown(answer)
            
            if sources:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}:** {source}")
    
    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

# Empty state
if not st.session_state.messages:
    if documents:
        st.markdown(f"""
        <div style="text-align: center; padding: 50px; color: #6b7280;">
            <h3>💬 Ready to Chat!</h3>
            <p>Selected: <strong>{st.session_state.selected_doc_name}</strong></p>
            <p>Ask any question about your document below.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 50px; color: #6b7280;">
            <h3>👋 Welcome!</h3>
            <p>Upload a PDF document using the sidebar to get started.</p>
            <br>
            <p><strong>Features:</strong></p>
            <ul style="list-style: none; padding: 0;">
                <li>� Upload multiple documents</li>
                <li>� Switch between documents anytime</li>
                <li>💬 Chat with context and history</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
