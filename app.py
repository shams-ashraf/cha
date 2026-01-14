# This is the main Streamlit app that runs the chatbot UI
# Flow: User opens app → loads existing ChromaDB → user asks question → answer_question() → display result

import streamlit as st
import uuid
import chromadb
import os
import re
import time
from styles import load_custom_css
from chromadb.utils import embedding_functions
from API import answer_question

# Configure Streamlit page
st.set_page_config(
    page_title="Biomedical Document Chatbot",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_custom_css()

# Folder paths
CHROMA_FOLDER = "./chroma_db"

# Create folder if it doesn't exist
os.makedirs(CHROMA_FOLDER, exist_ok=True)

# Get embedding function for ChromaDB
def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large"
    )

# Initialize session state
if "collection" not in st.session_state:
    st.session_state.collection = None
if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.active_chat = None

###########################################################################
# LOAD EXISTING CHROMADB
###########################################################################

try:
    client = chromadb.PersistentClient(path=CHROMA_FOLDER)
    collections = client.list_collections()
    
    if collections:
        # Load existing collection
        collection = client.get_collection(
            name=collections[0].name,
            embedding_function=get_embedding_function()
        )
        st.session_state.collection = collection
        
        # Get collection info
        collection_count = collection.count()
    else:
        st.error("❌ No ChromaDB collection found! Please run the document processing first.")
        st.info("💡 Make sure you have a ChromaDB in the './chroma_db' folder")
        st.stop()
        
except Exception as e:
    st.error(f"❌ Error loading ChromaDB: {str(e)}")
    st.info("💡 Make sure the ChromaDB exists in './chroma_db' folder")
    st.stop()

###########################################################################
# CHAT INTERFACE SECTION
###########################################################################

# Initialize first chat if none exists
if not st.session_state.chats:
    cid = f"chat_{uuid.uuid4().hex[:6]}"
    st.session_state.chats[cid] = {
        "title": "New Chat",
        "messages": [],
        "context": []
    }
    st.session_state.active_chat = cid

# Display header
st.markdown("""
<div class="main-card">
    <h1 style='text-align:center;margin:0;'>🧬 Biomedical Document Chatbot</h1>
</div>

<div class="main-card">
    <p style="text-align:center; font-size:1.1rem;">Answers <strong>only</strong> from official documents • Supports English & German • Remembers conversation</p>
    <h3 style="color:#00d9ff;">Try these examples:</h3>
    <ul style="font-size:1.05rem;">
        <li>What are the requirements for registering the master's thesis?</li>
        <li>Was sind die Regelungen fur die Masterarbeit?</li>
    </ul>
</div>
""", unsafe_allow_html=True)

###########################################################################
# SIDEBAR: Chat management
###########################################################################

with st.sidebar:
    st.markdown("# 🧬 BioMed Chat")
    
    # New chat button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        cid = f"chat_{uuid.uuid4().hex[:6]}"
        st.session_state.chats[cid] = {
            "title": "New Chat",
            "messages": [],
            "context": []
        }
        st.session_state.active_chat = cid
        st.rerun()
    
    st.markdown("### 💬 Your Chats")
    
    # Display all chats (newest first)
    for cid in reversed(list(st.session_state.chats.keys())):
        chat = st.session_state.chats[cid]
        col1, col2 = st.columns([4, 1])
        
        # Chat button
        with col1:
            if st.button(f"💬 {chat['title'][:35]}...", key=f"open_{cid}", use_container_width=True):
                st.session_state.active_chat = cid
                st.rerun()
        
        # Delete button
        with col2:
            if st.button("🗑️", key=f"del_{cid}"):
                del st.session_state.chats[cid]
                if st.session_state.active_chat == cid:
                    st.session_state.active_chat = next(iter(st.session_state.chats), None)
                st.rerun()

###########################################################################
# MAIN CHAT AREA
###########################################################################

# Get current chat
chat = st.session_state.chats[st.session_state.active_chat]

# Display chat history
for m in chat["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
if query := st.chat_input("Ask anything about the MBE program documents..."):
    # Add user message
    chat["messages"].append({"role": "user", "content": query})
    
    # Update chat title if it's new
    if chat["title"] == "New Chat":
        chat["title"] = query[:40] + "..." if len(query) > 40 else query
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents & thinking..."):
            # Call answer function
            answer, used_chunks = answer_question(
                query,
                chat["messages"],
                collection=st.session_state.collection
            )
            
            st.markdown(answer, unsafe_allow_html=True)
            
            # Handle rate limit countdown
            match = re.search(r'wait (\d+) seconds', answer.lower())
            if match:
                remaining = int(match.group(1))
                countdown = st.empty()
                while remaining > 0:
                    countdown.warning(f"⏳ Please wait {remaining} seconds before sending a new request...")
                    time.sleep(1)
                    remaining -= 1
                countdown.success("✅ You can now send a new question.")
    
    # Add assistant message
    chat["messages"].append({"role": "assistant", "content": answer})
    chat["context"] = used_chunks if used_chunks else []
