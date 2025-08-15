import streamlit as st
import tempfile
import os
from rag_app import RAGApp
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_TEMPERATURE, DEFAULT_K_RETRIEVAL


def initialize_rag_app():
    """Initialize the RAG application."""
    try:
        # Set up event loop for initialization
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        rag_app = RAGApp(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            temperature=DEFAULT_TEMPERATURE,
            k_retrieval=DEFAULT_K_RETRIEVAL
        )
        return rag_app
    except Exception as e:
        st.error(f"Error initializing RAG app: {e}")
        return None


def render_document_upload():
    """Render the document upload section."""
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose document files",
        type=['docx', 'pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload multiple documents to add to the knowledge base"
    )
    
    if uploaded_files:
        if st.button("📥 Process Uploaded Documents"):
            if st.session_state.rag_app:
                with st.spinner("Processing documents..."):
                    for uploaded_file in uploaded_files:
                        try:
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_path = tmp_file.name
                            
                            # Load the document
                            result = st.session_state.rag_app.load_documents(tmp_file_path)
                            st.success(f"✅ {uploaded_file.name}: {result}")
                            
                            # Clean up temporary file
                            os.unlink(tmp_file_path)
                            
                        except Exception as e:
                            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            else:
                st.error("RAG app not initialized. Please wait for initialization to complete.")


def render_document_management():
    """Render the document management section."""
    st.subheader("Manage Documents")
    
    # Show collection info
    if st.session_state.rag_app:
        try:
            collection_info = st.session_state.rag_app.get_collection_info()
            if "error" not in collection_info:
                st.info(f"📊 Index: {collection_info['name']}")
                st.info(f"📄 Vectors: {collection_info['count']} total")
                
                # Show documents list if there are documents
                if collection_info['count'] > 0:
                    st.subheader("📋 Current Documents")
                    documents = st.session_state.rag_app.get_documents_list()
                    if documents:
                        for i, doc in enumerate(documents[:5]):  # Show first 5 documents
                            with st.expander(f"Document {i+1}: {doc['id']}"):
                                st.text(doc['content'])
                                if doc['metadata']:
                                    st.caption(f"Metadata: {doc['metadata']}")
                        if len(documents) > 5:
                            st.caption(f"... and {len(documents) - 5} more documents")
                    else:
                        st.caption("No document details available")
            else:
                st.warning("No collection information available")
        except Exception as e:
            st.warning(f"Could not retrieve collection info: {str(e)}")
    
    # Clear documents button
    if st.button("🗑️ Clear All Vectors", type="secondary"):
        if st.session_state.rag_app:
            with st.spinner("Clearing documents..."):
                try:
                    result = st.session_state.rag_app.clear_documents()
                    st.success(result)
                    st.rerun()  # Refresh the page to update collection info
                except Exception as e:
                    st.error(f"Error clearing documents: {str(e)}")
        else:
            st.error("RAG app not initialized. Please wait for initialization to complete.")


def render_sidebar():
    """Render the sidebar with document management features."""
    # Check authentication first
    from auth import authenticate_user
    
    if authenticate_user():
        # User is authenticated - show document management features
        st.header("📚 Document Management")
        
        # Logout button
        if st.button("🚪 Logout", type="secondary"):
            st.session_state.authenticated = False
            st.rerun()
        
        st.divider()
        
        # Document upload section
        render_document_upload()
        
        st.divider()
        
        # Document management section
        render_document_management()
        
        st.divider()
        
        # Refresh button
        if st.button("🔄 Refresh Document List"):
            st.rerun()


def render_chat_interface():
    """Render the main chat interface."""
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask anything about the Cogent Labs policies..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            try:
                # Get last 5 messages for conversation history
                conversation_history = ""
                if len(st.session_state.messages) > 1:  # More than just the current user message
                    # Get last 5 messages (excluding the current one)
                    recent_messages = st.session_state.messages[-6:-1]  # Last 5 before current
                    conversation_history = "\n".join([
                        f"{msg['role'].title()}: {msg['content']}" 
                        for msg in recent_messages
                    ])
                
                # Initialize full response for chat history
                full_response = ""
                first_chunk_received = False
                
                # Show thinking spinner initially
                with st.spinner("Thinking..."):
                    # Stream the response
                    for chunk in st.session_state.rag_app.query_stream(prompt, conversation_history):
                        if not first_chunk_received:
                            # First chunk received, clear spinner and start streaming
                            first_chunk_received = True
                            st.empty()  # Clear the spinner
                        
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                
                # Display final response without cursor
                message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
            except Exception as e:
                error_message = f"Error processing your question: {str(e)}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
