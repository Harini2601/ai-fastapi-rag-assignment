<<<<<<< HEAD
# Day 3 Assignment – Retrieval Augmented Generation (RAG) with FastAPI

This project implements a complete RAG pipeline using FastAPI and LangChain (offline mode).

##  Features
- Upload documents (PDF/TXT)
- Chunking using Recursive & Simple strategies
- Vector database using ChromaDB
- Query documents using RetrievalQA
- Offline LLM and Embeddings
- CRUD APIs for documents

##  Endpoints

| Endpoint | Description |
|--------|------------|
| POST /upload | Upload document |
| POST /query | Ask question |
| GET /documents | List documents |
| DELETE /documents/{id} | Delete document |

##  Tech Stack
- Python 3.11
- FastAPI
- LangChain
- ChromaDB

##  Run
pip install fastapi uvicorn langchain langchain-community langchain-text-splitters chromadb pypdf python-multipart
uvicorn main:app --reload
=======

