#!/usr/bin/env python
# coding: utf-8

# In[4]:
# import os;
# os.system('pip install -r requirements.txt')

import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama.chat_models import ChatOllama
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document
from rank_bm25 import BM25Okapi
from typing import List
import sys
# import pysqlite3

# sys.modules["sqlite3"] = pysqlite3  # Override default sqlite3 with pysqlite3
import streamlit as st
import ollama

# Streamlit UI
st.title("Ollama Chatbot with Streamlit")

# User input
user_input = st.text_input("Ask a question:")

if user_input:
    response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": user_input}])
    st.write("**Response:**", response['message']['content'])




# In[5]:


def read_pdfs(pdf_directory: str) -> List[dict]:
    """Reads all PDFs from a directory and returns a list of documents with text and metadata."""
    all_documents = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_directory, filename)
            try:
                loader = PyMuPDFLoader(filepath)
                documents = loader.load()
                for doc in documents:
                    # Ensure metadata has 'source'
                    if 'source' not in doc.metadata:
                        doc.metadata["source"] = filename
                    all_documents.append({"text": doc.page_content, "metadata": doc.metadata})
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return all_documents

def chunk_text(documents: List[dict], chunk_size: int = 100, chunk_overlap: int = 0) -> List[dict]:
    """Splits a list of documents into chunks with metadata."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for document in documents:
        chunks = text_splitter.split_text(document["text"])
        for chunk in chunks:
            all_chunks.append({"text": chunk, "metadata": document["metadata"]})
    return all_chunks

def store_chunks_in_chroma(chunks: List[dict], embeddings, persist_directory: str = "db"):
    """Stores text chunks with metadata in Chroma vector database."""
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    vectordb = Chroma.from_texts(texts, embeddings, metadatas=metadatas, persist_directory=persist_directory)
    return vectordb

def create_bm25_index(chunks: List[dict]) -> BM25Retriever:
    """Creates a BM25 index from document chunks."""
    # Convert chunk dictionaries to Document objects
    documents = [
        Document(page_content=chunk["text"], metadata=chunk["metadata"])
        for chunk in chunks
    ]
    
    # Tokenize document page_content for BM25
    tokenized_docs = [doc.page_content.split() for doc in documents]
    bm25_index = BM25Okapi(tokenized_docs)
    
    # Wrap the BM25 index in a LangChain-compatible retriever using the correct keyword
    bm25_retriever = BM25Retriever(vectorizer=bm25_index, docs=documents)
    return bm25_retriever

# --- Main execution ---
# pdf_dir = "data/docs"  # Path to your PDF directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_dir = os.path.join(BASE_DIR, "data", "docs")
persist_dir = "db"     # Directory to store Chroma DB

# 1. Read PDFs and create chunks (do this once)
pdf_documents = read_pdfs(pdf_dir)
chunks = chunk_text(pdf_documents, chunk_size=500, chunk_overlap=50)

# 2. Load or create the Chroma vector store using the same chunks
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
if os.path.exists(persist_dir) and os.listdir(persist_dir):
    print("Loading existing Chroma database...")
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
else:
    print("Chroma database not found. Processing PDFs...")
    vectordb = store_chunks_in_chroma(chunks, embeddings, persist_dir)

# 3. Create BM25 index from the already created chunks (used for keyword search)
bm25_retriever = create_bm25_index(chunks)

# 4. Initialize language model
llm = ChatOllama(
    model="tinyllama",
    temperature=0.7,
    base_url="http://localhost:11434"
)

# 5. Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key='answer'
)

# 6. Create an ensemble retriever combining vector DB and BM25 retrievers
vector_retriever = vectordb.as_retriever()
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.6, 0.4]  # Adjust weights based on your use case
)

# 7. Create conversation chain with memory and ensemble retriever
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=ensemble_retriever,
    memory=memory,
    return_source_documents=True,
    chain_type="stuff"
)




# Test question

# In[4]:



# CLI interface for user input

# In[7]:


while True:
    user_input = input("\nYour question (type 'exit' to quit): ")
    if user_input.lower() in ['exit', 'quit']:
        break
    result = qa_chain.invoke({"question": user_input})
    print(f"\nQuestion: {result['question']}")
    print(f"\nANSWER: {result['answer']}")
    
    if result['source_documents']:
        print("\nReference Documents:")
        for doc in result['source_documents'][:3]: 
            source = doc.metadata.get('source', 'Unknown source')
            page = doc.metadata.get('page', 'N/A')
            print(f"- {source} (page {page})")

