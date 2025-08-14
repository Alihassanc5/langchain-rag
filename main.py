import streamlit as st
import hashlib
import hmac

from typing import List, Dict, Any
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class RAGApp:
    """
    A RAG (Retrieval-Augmented Generation) application for document-based question answering.
    """
    
    def __init__(self, 
                 collection_name: str = "policies",
                 persist_directory: str = "./chroma_langchain_db",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 embedding_model: str = "models/gemini-embedding-001",
                 llm_model: str = "gemini-2.5-flash",
                 temperature: float = 0.0,
                 k_retrieval: int = 3):
        """
        Initialize the RAG application.
        
        Args:
            collection_name: Name for the vector store collection
            persist_directory: Directory to persist the vector store
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between text chunks
            embedding_model: Model name for embeddings
            llm_model: Model name for the language model
            temperature: Temperature for the language model
            k_retrieval: Number of documents to retrieve for each query
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_retrieval = k_retrieval
        
        # Initialize components
        self._setup_embeddings(embedding_model)
        self._setup_vector_store()
        self._setup_llm(llm_model, temperature)
        self._setup_prompt()
        self._setup_graph()
        
    def _setup_embeddings(self, embedding_model: str):
        """Initialize the embedding model."""
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    
    def _setup_vector_store(self):
        """Initialize the vector store."""
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
    
    def _setup_llm(self, llm_model: str, temperature: float):
        """Initialize the language model."""
        self.model = init_chat_model(
            model=llm_model, 
            model_provider="google_genai", 
            temperature=temperature
        )
    
    def _setup_prompt(self):
        """Initialize the RAG prompt."""
        # self.prompt = hub.pull("rlm/rag-prompt")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions using related to cogent lab policies. Don't tell the user about techincal terms or jargon."),
            ("human", "Context:\n{context}\n\nConversation History:\n{conversation_history}\n\nQuestion: {question}")
        ])
    
    def _setup_graph(self):
        """Initialize the LangGraph workflow."""
        graph_builder = StateGraph(State).add_sequence([self._retrieve, self._generate])
        graph_builder.add_edge(START, "_retrieve")
        self.graph = graph_builder.compile()
    
    def load_documents(self, file_path: str) -> str:
        """
        Load and process documents from a file.
        
        Args:
            file_path: Path to the document file
        """
        try:
            # Determine file type and load appropriate loader
            file_extension = file_path.lower().split('.')[-1]
            
            if file_extension == 'docx':
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                raise Exception(f"Unsupported file type: {file_extension}")
            
            # Load documents
            docs = loader.load()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            all_splits = text_splitter.split_documents(docs)
            
            # Add to vector store
            self.vector_store.add_documents(documents=all_splits)
            
            return f"Successfully loaded {len(all_splits)} document chunks from {file_path}"
            
        except Exception as e:
            raise Exception(f"Error loading documents: {e}")
    
    def _retrieve(self, state: State) -> Dict[str, List[Document]]:
        """Retrieve relevant documents for the question."""
        retrieved_docs = self.vector_store.similarity_search(
            state["question"], 
            k=self.k_retrieval
        )
        return {"context": retrieved_docs}
    
    def _generate(self, state: State) -> Dict[str, str]:
        """Generate answer based on retrieved context."""
        docs_content = "\n\n".join([doc.page_content for doc in state["context"]])
        
        # Get conversation history from state if available
        conversation_history = state.get("conversation_history", "")
        
        messages = self.prompt.invoke({
            "question": state["question"], 
            "context": docs_content,
            "conversation_history": conversation_history
        })
        print(messages)
        response = self.model.invoke(messages)
        return {"answer": response.content}
    
    def query(self, question: str, conversation_history: str = "") -> str:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
            conversation_history: Previous conversation messages
            
        Returns:
            The generated answer
        """
        try:
            result = self.graph.invoke({
                "question": question,
                "conversation_history": conversation_history
            })
            print("History: ", conversation_history)
            return result["answer"]
        except Exception as e:
            raise Exception(f"Error processing query: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector store collection."""
        collection = self.vector_store._collection
        if collection:
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
        return {"error": "Collection not found"}
    
    def get_documents_list(self) -> List[Dict[str, Any]]:
        """Get a list of documents in the collection with their metadata."""
        try:
            collection = self.vector_store._collection
            if collection:
                # Get all documents from the collection
                results = collection.get()
                documents = []
                
                if results and results['documents']:
                    for i, doc in enumerate(results['documents']):
                        metadata = results['metadatas'][i] if results['metadatas'] else {}
                        documents.append({
                            "content": doc[:100] + "..." if len(doc) > 100 else doc,
                            "metadata": metadata,
                            "id": results['ids'][i] if results['ids'] else f"doc_{i}"
                        })
                
                return documents
            return []
        except Exception as e:
            return []
    
    def clear_documents(self) -> str:
        """Clear all documents from the vector store."""
        try:
            self.vector_store.delete_collection()
            return "All documents cleared from the vector store."
        except Exception as e:
            raise Exception(f"Error clearing documents: {e}")


# Authentication functions
def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(password: str, hashed_password: str) -> bool:
    """Check if a password matches the hashed password."""
    return hmac.compare_digest(hash_password(password), hashed_password)

def init_authentication():
    """Initialize authentication settings."""
    # Set default admin credentials if not already set
    if "admin_password_hash" not in st.session_state:
        # Default password: "admin123" - change this in production!
        default_password = "admin123"
        st.session_state.admin_password_hash = hash_password(default_password)
    
    # Initialize authentication state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

def authenticate_user():
    """Authenticate user for admin access."""
    if not st.session_state.authenticated:
        st.sidebar.header("🔐 Admin Authentication")
        
        # Password input
        password = st.sidebar.text_input(
            "Enter Admin Password",
            type="password",
            help="Enter the admin password to access document management features"
        )
        
        # Login button
        if st.sidebar.button("🔑 Login"):
            if check_password(password, st.session_state.admin_password_hash):
                st.session_state.authenticated = True
                st.sidebar.success("✅ Authentication successful!")
                st.rerun()
            else:
                st.sidebar.error("❌ Invalid password!")
                
        return False
    
    return True

# Streamlit App
def main():
    # Page configuration
    st.set_page_config(
        page_title="RAG Chat Assistant",
        page_icon="🤖",
        layout="wide"
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "rag_app" not in st.session_state:
        st.session_state.rag_app = None

    # Initialize authentication
    init_authentication()

    # Title and description
    st.title("Cogent Lab Policies Assistant")
    st.markdown("Ask questions about the Cogent Lab policies.")

    # Sidebar for document management with authentication
    with st.sidebar:
        # Check authentication first
        if authenticate_user():
            # User is authenticated - show document management features
            st.header("📚 Document Management")
            
            # Logout button
            if st.button("🚪 Logout", type="secondary"):
                st.session_state.authenticated = False
                st.rerun()
            
            st.divider()
            
            # Document upload section
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
                                    import tempfile
                                    import os
                                    
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
            
            st.divider()
            
            # Document management section
            st.subheader("Manage Documents")
            
            # Show collection info
            if st.session_state.rag_app:
                try:
                    collection_info = st.session_state.rag_app.get_collection_info()
                    if "error" not in collection_info:
                        st.info(f"📊 Collection: {collection_info['name']}")
                        st.info(f"📄 Documents: {collection_info['count']} chunks")
                        
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
            if st.button("🗑️ Clear All Documents", type="secondary"):
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
            
            st.divider()
            
            # Refresh button
            if st.button("🔄 Refresh Document List"):
                st.rerun()
        else:
            # User is not authenticated - authentication form is shown by authenticate_user()
            pass
        
    # Initialize RAG app
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
                collection_name="policies",
                persist_directory="./chroma_langchain_db",
                chunk_size=1500,
                chunk_overlap=100,
                temperature=0.0,
                k_retrieval=3
            )
            return rag_app
        except Exception as e:
            st.error(f"Error initializing RAG app: {e}")
            return None

    # Initialize RAG app if not already done
    if st.session_state.rag_app is None:
        with st.spinner("Loading..."):
            st.session_state.rag_app = initialize_rag_app()
            # st.session_state.rag_app.load_documents("food.docx")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask anything about the Cogent Lab policies..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            try:
                with st.spinner("Thinking..."):
                    # Get last 5 messages for conversation history
                    conversation_history = ""
                    if len(st.session_state.messages) > 1:  # More than just the current user message
                        # Get last 5 messages (excluding the current one)
                        recent_messages = st.session_state.messages[-6:-1]  # Last 5 before current
                        conversation_history = "\n".join([
                            f"{msg['role'].title()}: {msg['content']}" 
                            for msg in recent_messages
                        ])
                    
                    # Run the query synchronously with conversation history
                    answer = st.session_state.rag_app.query(prompt, conversation_history)
                    
                    message_placeholder.markdown(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
            except Exception as e:
                error_message = f"Error processing your question: {str(e)}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()
