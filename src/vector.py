from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

db_location = "./chroma_langchain_db"

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = Chroma(
    collection_name="irc_code",
    embedding_function=embeddings,
    persist_directory = db_location
)

add_documents = not os.path.exists(db_location)
if add_documents:
    file_path = "./resources/(3.13.25) IRS CODE.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )

    all_splits = text_splitter.split_documents(docs)

    ids = vector_store.add_documents(documents=all_splits)

retriever = vector_store.as_retriever()