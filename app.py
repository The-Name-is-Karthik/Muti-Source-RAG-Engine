import streamlit as st
import os
from langchain.schema.document import Document

from src.video_processor import get_video_transcript
from src.data_loader import load_from_webpage, load_from_pdf, load_from_docx
from src.vector_store import create_vector_store
from src.rag_pipeline import create_rag_chain

# --- Page Configuration ---
st.set_page_config(page_title="Muti-Source-RAG-Engine", layout="wide")

# --- Application State Management ---
# --- Using  a dictionary for separate chat histories ---
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {
        "url_chat": [],
        "doc_chat": []
    }
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
# --- Keeping  track of the last processed source to prevent re-embedding ---
if "last_processed_source" not in st.session_state:
    st.session_state.last_processed_source = ""

# --- UI Rendering ---
st.title("Multi-Source RAG Engine.")
st.info("Ask questions about YouTube videos, web pages, or your own documents!")

# --- UI with Tabs for different data sources ---
tab1, tab2 = st.tabs(["URL (YouTube or Web Page)", "Upload Document (PDF/DOCX)"])

# --- URL Tab ---
with tab1:
    st.header("Process a URL")
    url = st.text_input("Enter any URL (YouTube, blog, news, etc.):", key="url_input")
    
    if st.button("Process URL"):
        if url:
            # --- Only process if the URL is new ---
            if url != st.session_state.last_processed_source:
                with st.spinner("Processing URL... This may take a moment."):
                    documents = []
                    if "youtube.com" in url or "youtu.be" in url:
                        transcript = get_video_transcript(url)
                        if transcript:
                            documents = [Document(page_content=transcript)]
                    else:
                        documents = load_from_webpage(url)
                    
                    if documents:
                        vector_store = create_vector_store(documents)
                        st.session_state.rag_chain = create_rag_chain(vector_store.as_retriever())
                        st.session_state.last_processed_source = url # Mark as processed
                        st.session_state.chat_histories["url_chat"] = [] # Clear chat
                        st.success("URL processed! You can now ask questions below.")
                    else:
                        st.error("Could not retrieve any content from the URL.")
            else:
                st.toast("This URL has already been processed.")
        else:
            st.warning("Please enter a URL.")

    # --- Display and manage the URL-specific chat ---
    for message in st.session_state.chat_histories["url_chat"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the URL...", key="url_chat_input"):
        st.session_state.chat_histories["url_chat"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if st.session_state.rag_chain:
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke(prompt)
                st.session_state.chat_histories["url_chat"].append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
        else:
            st.warning("Please process a URL first.")

# --- Document Upload Tab ---
with tab2:
    st.header("Process a Document")
    uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])
    
    if uploaded_file is not None:
        # --- Use file_id and check if it's a new file ---
        doc_id = uploaded_file.file_id
        if doc_id != st.session_state.last_processed_source:
            with st.spinner("Processing document..."):
                temp_dir = "temp_docs"
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                documents = []
                if uploaded_file.type == "application/pdf":
                    documents = load_from_pdf(temp_path)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    documents = load_from_docx(temp_path)
                
                vector_store = create_vector_store(documents)
                st.session_state.rag_chain = create_rag_chain(vector_store.as_retriever())
                st.session_state.last_processed_source = doc_id # Mark as processed
                st.session_state.chat_histories["doc_chat"] = [] # Clear chat
                st.success("Document processed! You can now ask questions below.")
        
    # --- Display and manage the document-specific chat ---
    for message in st.session_state.chat_histories["doc_chat"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the document...", key="doc_chat_input"):
        st.session_state.chat_histories["doc_chat"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.rag_chain and st.session_state.last_processed_source == uploaded_file.file_id:
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke(prompt)
                st.session_state.chat_histories["doc_chat"].append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
        else:
            st.warning("Please upload and process a document first.")