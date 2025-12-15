# RAG Bot Backend

This repository contains the backend implementation for a Retrieval-Augmented Generation (RAG) bot. The bot uses vector databases, language models, and document ingestion pipelines to provide intelligent query responses based on ingested documents and web searches.

## Features

- **Document Ingestion**: Upload documents or provide URLs to ingest content into the vector database.
- **Query Processing**: Perform intelligent queries using a combination of vector search and web search.
- **Session Management**: Generate unique session IDs for tracking user interactions.
- **Collection Management**: Create, list, and delete collections in the vector database.
- **Customizable Prompts**: Use predefined prompts for system and user interactions.
- **Exception Handling**: Robust error handling for ingestion, query, and database operations.

## Prerequisites

- Python 3.11
- Docker and Docker Compose (Optional)
- NVIDIA GPU with CUDA support (for GPU acceleration)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ShresthaAnkit/LangchainRAG.git
   cd LangchainRAG
   ```

2. **Set up the environment**:
   ```bash
    uv venv --python 3.11 .venv
    source .venv/bin/activate
    ```
   - Create a `.env` file in the `app/` directory with the required environment variables. Refer to `app/core/config.py` for the list of variables.

3. **Install dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Run the application locally**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

5. **Run with Docker**:
   - Build and start the services:
    ```bash
    docker compose up --build
    ```

   - The backend will be available at `http://localhost:8000`.

## API Endpoints

### Health Check
- **GET** `/api/health`
  - Check if the server is running.

### Session Management
- **GET** `/session`
  - Generate a unique session ID.

### Collection Management
- **POST** `/collection/{collection_name}`
  - Create a new collection.
- **GET** `/collection`
  - List all collections.
- **DELETE** `/collection/{collection_name}`
  - Delete a collection.

### Document Ingestion
- **POST** `/collection/{collection_name}/ingest-documents`
  - Upload documents for ingestion.
- **POST** `/collection/{collection_name}/ingest-urls`
  - Provide URLs for ingestion.

### Query Processing
- **POST** `/collection/{collection_name}/chat`
  - Perform a query using the ingested data.

## Configuration

The application uses environment variables for configuration. These can be set in the `.env` file located in the `app/` directory. Key settings include:

- **Vector Database**:
  - `VECTOR_DB`: The vector database to use (e.g., `qdrant`).
  - `VECTORDB_PERSIST_URL`: URL for the vector database.

- **Language Model**:
  - `LLM_PROVIDER`: The language model provider (e.g., `google`, `cohere`, `bedrock`).
  - `LLM_MODEL_NAME`: The model name to use.

- **Redis**:
  - `REDIS_URL`: Redis URL for session history.

- **API Keys**:
  - `GOOGLE_API_KEY`, `COHERE_API_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, etc.

## Logging

Logs are configured to use the `Asia/Kathmandu` timezone. The log level can be set using the `LOG_LEVEL` environment variable.