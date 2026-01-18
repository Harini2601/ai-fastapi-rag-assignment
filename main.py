# main.py
"""
Day 3 Assignment: Retrieval-Augmented Generation (RAG) with FastAPI + LangChain (Offline)

Endpoints:
- POST   /upload        -> Upload documents (PDF or TXT)
- POST   /query         -> Ask questions over uploaded documents (RAG)
- GET    /documents     -> List uploaded documents
- DELETE /documents/{id}-> Delete a document

Features:
- Embeddings + Vector DB (Chroma)
- Two chunking strategies: RecursiveCharacterTextSplitter & CharacterTextSplitter
- Document loaders (PDF, Text)
- RAG pipeline (Retriever + LLM)
- Offline Mock LLM (no API key required)

Run:
1) pip install fastapi uvicorn langchain langchain-community langchain-text-splitters chromadb pypdf python-multipart
2) uvicorn main:app --reload
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import os, uuid, shutil

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import asyncio

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Day 3: FastAPI RAG System (Offline)")

UPLOAD_DIR = "uploads"
DB_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# ----------------------------
# Offline Embeddings & LLM
# ----------------------------
class MockEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(len(t))] * 384 for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return [float(len(text))] * 384

class MockLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "mock-llm"

    async def _acall(self, prompt: str, stop=None, **kwargs) -> str:
        await asyncio.sleep(0.1)
        return "This is an offline RAG answer generated from retrieved documents."

# ----------------------------
# Models
# ----------------------------
class QueryRequest(BaseModel):
    question: str
    chunking: str = "recursive"  # recursive | simple

class QueryResponse(BaseModel):
    answer: str

class DocumentInfo(BaseModel):
    id: str
    name: str

# ----------------------------
# Helpers
# ----------------------------

def get_splitter(strategy: str):
    if strategy == "simple":
        return CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# ----------------------------
# Endpoints
# ----------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Load document
    if file.filename.lower().endswith(".pdf"):
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    db = Chroma.from_documents(chunks, MockEmbeddings(), persist_directory=DB_DIR)
    db.persist()

    return {"message": "Document uploaded", "id": doc_id}

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    db = Chroma(persist_directory=DB_DIR, embedding_function=MockEmbeddings())
    retriever = db.as_retriever()

    prompt = PromptTemplate(
        template="Answer the question using the context.\nContext: {context}\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"],
    )

    qa = RetrievalQA.from_chain_type(
        llm=MockLLM(), retriever=retriever, chain_type_kwargs={"prompt": prompt}
    )

    answer = await qa.arun(req.question)
    return QueryResponse(answer=answer)

@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    files = []
    for f in os.listdir(UPLOAD_DIR):
        files.append(DocumentInfo(id=f.split("_")[0], name=f))
    return files

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    removed = False
    for f in os.listdir(UPLOAD_DIR):
        if f.startswith(doc_id):
            os.remove(os.path.join(UPLOAD_DIR, f))
            removed = True
    if not removed:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Deleted"}

# Run: uvicorn main:app --reload
