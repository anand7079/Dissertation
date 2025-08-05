import os
import streamlit as st
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.docstore.document import Document
from guardrails import Guard

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "data", "docs")
PERSIST_DIR = os.path.join(BASE_DIR, "db")

# --- Load PDFs ---
def load_pdf_files(pdf_folder: str) -> List[Document]:
    all_chunks = []
    if not os.path.exists(pdf_folder) or not os.listdir(pdf_folder):
        raise ValueError(f"No PDF files found in: {pdf_folder}")

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, filename))
            chunks = loader.load_and_split()
            all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No content loaded from PDF files.")
    return all_chunks

# --- Split Text into Chunks ---
def split_chunks(documents: List[Document]) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    return [{"text": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks]

# --- Cache Chroma Vector DB ---
@st.cache_resource
def get_chroma_db(chunks: List[dict], _embeddings, persist_directory: str):
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=_embeddings)
    else:
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        vectordb = Chroma.from_texts(texts, _embeddings, metadatas=metadatas, persist_directory=persist_directory)
    return vectordb

# --- BM25 Retriever ---
def create_bm25_index(chunks: List[Document]) -> BM25Retriever:
    documents = [Document(page_content=chunk.page_content, metadata=chunk.metadata) for chunk in chunks]
    if not documents:
        raise ValueError("No documents found for BM25 index.")
    return BM25Retriever.from_documents(documents)

# --- Combine BM25 and Chroma ---
def get_qa_chain(pdf_dir: str, persist_dir: str):
    raw_docs = load_pdf_files(pdf_dir)
    chunks = split_chunks(raw_docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = get_chroma_db(chunks, embeddings, persist_dir)
    chroma_retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    from langchain.docstore.document import Document
    chunks = [Document(page_content=chunk["text"], metadata=chunk["metadata"]) for chunk in chunks]


    bm25_retriever = create_bm25_index(chunks)
    bm25_retriever.k = 3

    # Combine both retrievers
    from langchain.retrievers.ensemble import EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(retrievers=[chroma_retriever, bm25_retriever], weights=[0.5, 0.5])

    llm = Ollama(model="tinyllama")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=ensemble_retriever)
    return qa

# --- CLI Mode ---
def run_cli_mode(qa_chain):
    print("Chatbot CLI (type 'exit' to quit)")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = qa_chain.run(query)
        print("Bot:", answer)
# --- Guardrails ---
from guardrails import Guard

guard = Guard.for_rail("guardrails_spec.rail")

def run_query_with_guardrails(user_query):
    # Retrieve context
    docs = qa_chain.retriever.get_relevant_documents(user_query)
    context = "\n\n".join([d.page_content for d in docs])

    # Run your QA LLM chain
    raw_output = qa_chain.run(user_query)

    # Validate without messages or api
    validation_result = guard.parse(
        llm_output=raw_output,
        prompt_params={"question": user_query, "context": context}
    )

    return validation_result.validated_output


# --- Streamlit UI ---
def run_streamlit_ui(qa_chain):
    st.title("PDF Chatbot 🤖📄")
    user_input = st.text_input("Ask a question:")

    if user_input:
        result = run_query_with_guardrails(user_input)
        st.markdown("### Answer:")
        st.write(result)

# --- Entry Point ---
if __name__ == "__main__":
    import sys
    qa_chain = get_qa_chain(PDF_DIR, PERSIST_DIR)

    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli_mode(qa_chain)
    else:
        run_streamlit_ui(qa_chain)
