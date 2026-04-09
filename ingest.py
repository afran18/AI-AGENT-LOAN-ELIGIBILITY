import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from config import embedding_model

def ingest_policy(file_path: str):
    # 1. Load the PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )

    print(f"Ingested {len(splits)} chunks into ChromaDB.")
    return vectorstore

if __name__ == "__main__":
    PDF_PATH = "data/BankLoanPolicy_Rulebook.pdf"
    if os.path.exists(PDF_PATH):
        ingest_policy(PDF_PATH)
    else:
        print(f"Could not find the PDF {PDF_PATH}")