"""
RAG Pipeline for Offline Personal Notes Search
----------------------------------------------

This module handles:
1. Loading PDF/TXT documents
2. Splitting documents into chunks
3. Creating a FAISS vector database
4. Running a RAG (Retrieval-Augmented Generation) pipeline
   using local offline models through Ollama.

Works fully offline with:
- Embeddings: all-minilm:33m (Ollama)
- LLM Model: qwen2:1.5b (Ollama)

Author: Devdatta Shelar
GitHub-ready version
"""

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


# -------------------------------------------------------------------
# Global Paths (GitHub Friendly — relative, not absolute)
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAISS_PATH = BASE_DIR / "faiss_db"


# -------------------------------------------------------------------
# Load documents
# -------------------------------------------------------------------
def load_documents(folder_path=DATA_DIR):
    """
    Load all PDF and TXT documents from the given folder.
    Returns a list of LangChain Document objects.
    """
    docs = []

    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"[WARNING] Data folder not found: {folder_path}")
        return []

    for file in folder_path.iterdir():
        if file.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(file)).load())
        elif file.suffix.lower() == ".txt":
            docs.extend(TextLoader(str(file)).load())

    print(f"[INFO] Loaded {len(docs)} documents.")
    return docs


# -------------------------------------------------------------------
# Split documents into manageable chunks
# -------------------------------------------------------------------
def split_documents(docs):
    """
    Splits documents into smaller chunks for vector search.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunks.")
    return chunks


# -------------------------------------------------------------------
# Create vector database (FAISS)
# -------------------------------------------------------------------
def create_vector_db(chunks):
    """
    Creates and saves the FAISS vector database.
    """
    embeddings = OllamaEmbeddings(model="all-minilm:33m")

    vdb = FAISS.from_documents(chunks, embeddings)

    FAISS_PATH.mkdir(exist_ok=True)
    vdb.save_local(str(FAISS_PATH))

    print(f"[INFO] FAISS DB created at: {FAISS_PATH}")
    return vdb


# -------------------------------------------------------------------
# Build RAG QA Pipeline
# -------------------------------------------------------------------
def get_rag_qa():
    """
    Loads FAISS DB and creates a complete RAG pipeline.
    Returns a callable RAG chain: rag_chain.invoke(question)
    """

    embeddings = OllamaEmbeddings(model="all-minilm:33m")

    # Load FAISS DB
    vdb = FAISS.load_local(
        str(FAISS_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )

    print(f"🔥 FAISS index loaded. Dimension = {vdb.index.d}")

    retriever = vdb.as_retriever(search_kwargs={"k": 2})

    llm = ChatOllama(
        model="qwen2:1.5b",
        temperature=0
    )

    # Prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Use ONLY the following notes to answer.\n\n"
            "Notes:\n{context}\n\n"
            "Question: {question}\n"
            "Answer clearly."
        )
    )

    # LCEL RAG Chain
    rag_chain = (
        RunnableParallel(
            context=retriever,
            question=RunnablePassthrough()
        )
        | (lambda x: {
            "context": "\n\n".join(
                [d.page_content for d in x["context"]]
            )[:1400],   # limit context to keep speed fast
            "question": x["question"]
        })
        | prompt
        | llm
    )

    return rag_chain
