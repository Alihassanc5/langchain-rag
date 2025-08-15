# LangChain RAG with Pinecone

A Retrieval-Augmented Generation (RAG) application built with LangChain and Pinecone for document-based question answering.

## Features

- Document upload and processing (DOCX, PDF, TXT)
- Vector storage using Pinecone
- Chat interface for querying documents
- Admin authentication for document management
- Conversation history support

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the root directory with the following variables:

```env
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
PINECONE_INDEX_NAME=policies

# Google AI Configuration
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Pinecone Setup

1. Sign up for a Pinecone account at [pinecone.io](https://pinecone.io)
2. Create a new project and get your API key
3. Note your environment (e.g., `us-east-1-aws`)
4. Add these credentials to your `.env` file

### 4. Google AI Setup

1. Get a Google AI API key from [Google AI Studio](https://aistudio.google.com/)
2. Add the API key to your `.env` file

## Usage

### Running the Application

```bash
streamlit run main.py
```

### Default Admin Credentials

- Username: admin
- Password: admin123

**Important**: Change these credentials in production!

### Features

- **Document Upload**: Upload documents through the admin interface
- **Chat Interface**: Ask questions about your uploaded documents
- **Vector Management**: View and manage your Pinecone index
- **Conversation History**: Maintain context across multiple questions

## Architecture

- **Vector Store**: Pinecone for scalable vector storage
- **Embeddings**: Google Gemini embeddings
- **LLM**: Google Gemini 2.5 Flash for text generation
- **Framework**: LangChain with LangGraph for workflow management
- **UI**: Streamlit for the web interface

## Notes

- The application automatically creates a Pinecone index named "policies" if it doesn't exist
- Document chunks are stored as vectors in Pinecone
- The system uses cosine similarity for vector search
- All vectors are stored in the cloud via Pinecone's managed service
- The index is configured for 768-dimensional vectors (compatible with Gemini embedding-001)
- If you encounter dimension mismatch errors, delete the existing index and let the app recreate it