# Cogent Labs Policies Assistant

A RAG (Retrieval-Augmented Generation) application for document-based question answering using LangChain, Pinecone, and Google's Gemini models.

## Project Structure

The application has been modularized into the following components:

```
langchain-rag/
├── main.py              # Main application entry point
├── config.py            # Configuration and environment variables
├── models.py            # Data models and type definitions
├── rag_app.py           # Core RAG application logic
├── auth.py              # Authentication functionality
├── ui_components.py     # Streamlit UI components
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Module Descriptions

- **`main.py`**: Entry point that orchestrates the application, handles page configuration, and initializes components
- **`config.py`**: Centralized configuration management for environment variables and default settings
- **`models.py`**: Type definitions and data structures used throughout the application
- **`rag_app.py`**: Core RAG functionality including document processing, vector storage, and query handling
- **`auth.py`**: User authentication logic for admin access to document management features
- **`ui_components.py`**: Streamlit UI components for chat interface and document management

## Features

- **Document Upload**: Upload and process Word documents (.docx) to the knowledge base
- **RAG-powered Q&A**: Ask questions about uploaded documents with context-aware responses
- **Streaming Responses**: Real-time streaming of AI responses for better user experience
- **Document Management**: View, manage, and clear documents from the vector store
- **Admin Authentication**: Secure access to document management features
- **Conversation History**: Maintains context across multiple questions

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**: Create a `.env` file with:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   PINECONE_INDEX_NAME=your_index_name
   ADMIN_PASSWORD=your_admin_password
   GOOGLE_API_KEY=your_google_api_key
   ```

3. **Run the Application**:
   ```bash
   streamlit run main.py
   ```

## Usage

1. **Access the Application**: Open your browser to the Streamlit URL (usually `http://localhost:8501`)

2. **Admin Authentication**: Click the sidebar and enter the admin password to access document management

3. **Upload Documents**: Use the sidebar to upload Word documents (.docx) to the knowledge base

4. **Ask Questions**: Use the chat interface to ask questions about the uploaded documents

5. **Manage Documents**: View document statistics and clear the knowledge base as needed

## Technical Details

- **Vector Store**: Pinecone for document storage and similarity search
- **Embeddings**: Google's Gemini embedding model
- **Language Model**: Google's Gemini 2.5 Flash for text generation
- **Framework**: LangChain for RAG pipeline orchestration
- **UI**: Streamlit for the web interface
- **Document Processing**: LangChain document loaders and text splitters

## Configuration

Key configuration options can be modified in `config.py`:

- `DEFAULT_CHUNK_SIZE`: Size of text chunks for document splitting (default: 1500)
- `DEFAULT_CHUNK_OVERLAP`: Overlap between text chunks (default: 100)
- `DEFAULT_K_RETRIEVAL`: Number of documents to retrieve per query (default: 3)
- `DEFAULT_TEMPERATURE`: LLM temperature for response generation (default: 0.0)

## Security

- Admin password is hashed using SHA-256
- Authentication state is managed through Streamlit session state
- Secure password comparison using `hmac.compare_digest`

## Dependencies

See `requirements.txt` for the complete list of Python packages required to run this application.