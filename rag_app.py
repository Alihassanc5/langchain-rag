from typing import List, Dict, Any
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
from langchain_core.prompts import ChatPromptTemplate

from config import (
    PINECONE_API_KEY, 
    PINECONE_ENVIRONMENT, 
    PINECONE_INDEX_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_K_RETRIEVAL
)
from models import State


class RAGApp:
    """
    A RAG (Retrieval-Augmented Generation) application for document-based question answering.
    """
    
    def __init__(self, 
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                 llm_model: str = DEFAULT_LLM_MODEL,
                 temperature: float = DEFAULT_TEMPERATURE,
                 k_retrieval: int = DEFAULT_K_RETRIEVAL):
        """
        Initialize the RAG application.
        
        Args:
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between text chunks
            embedding_model: Model name for embeddings
            llm_model: Model name for the language model
            temperature: Temperature for the language model
            k_retrieval: Number of documents to retrieve for each query
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_retrieval = k_retrieval
        self.index_name = PINECONE_INDEX_NAME
        self.pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        
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
        # Initialize Pinecone
        index = self.pc.Index(self.index_name)

        self.vector_store = PineconeVectorStore(
            index=index,
            embedding=self.embeddings,
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
            ("system", "You are a helpful assistant that answers questions using related to cogent labs policies. Don't tell the user about techincal terms or jargon."),
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
    
    def _generate_stream(self, state: State):
        """Generate streaming answer based on retrieved context."""
        docs_content = "\n\n".join([doc.page_content for doc in state["context"]])
        
        # Get conversation history from state if available
        conversation_history = state.get("conversation_history", "")
        
        messages = self.prompt.invoke({
            "question": state["question"], 
            "context": docs_content,
            "conversation_history": conversation_history
        })
        
        # Return streaming response
        return self.model.stream(messages)
    
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
    
    def query_stream(self, question: str, conversation_history: str = ""):
        """
        Query the RAG system with streaming response.
        
        Args:
            question: The question to ask
            conversation_history: Previous conversation messages
            
        Yields:
            Streaming chunks of the generated answer
        """
        try:
            # First retrieve relevant documents
            retrieved_docs = self.vector_store.similarity_search(
                question, 
                k=self.k_retrieval
            )
            
            # Then generate streaming response
            for chunk in self._generate_stream({"question": question, "context": retrieved_docs, "conversation_history": conversation_history}):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
        except Exception as e:
            yield f"Error processing query: {e}"
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the Pinecone index."""
        try:
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()
            return {
                "name": self.index_name,
                "count": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", 0),
                "metric": stats.get("metric", "cosine")
            }
        except Exception as e:
            return {"error": f"Index not found: {str(e)}"}
    
    def get_documents_list(self) -> List[Dict[str, Any]]:
        """Get a list of documents in the Pinecone index with their metadata."""
        try:
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()
            
            # Note: Pinecone doesn't provide a direct way to list all documents
            # This is a limitation of Pinecone's API
            documents = []
            if stats.get("total_vector_count", 0) > 0:
                documents.append({
                    "content": f"Index contains {stats.get('total_vector_count', 0)} vectors",
                    "metadata": {"note": "Pinecone doesn't provide direct document listing"},
                    "id": "pinecone_index"
                })
            
            return documents
        except Exception as e:
            return []
    
    def clear_documents(self) -> str:
        """Clear all documents from the Pinecone index."""
        try:
            index = self.pc.Index(self.index_name)
            # Delete all vectors in the index
            index.delete(delete_all=True)
            return "All documents cleared from the Pinecone index."
        except Exception as e:
            raise Exception(f"Error clearing documents: {e}")
