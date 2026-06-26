import os
import glob
from pathlib import Path
from datetime import datetime
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import create_engine, Column, Integer, Text, DateTime, MetaData, Table
from sqlalchemy.orm import Session
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OPENAI_API_KEY, CHROMA_PATH, SQLITE_DB_PATH

MODEL = "gpt-4.1-nano"

DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)


def fetch_documents():
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


def seed_sqlite():
    engine = create_engine(f"sqlite:///{SQLITE_DB_PATH}")
    metadata = MetaData()

    documents_table = Table(
        "documents",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("title", Text),
        Column("content", Text),
        Column("category", Text),
        Column("created_at", DateTime),
    )
    metadata.create_all(engine)

    sample_rows = [
        {
            "title": "Introduction to Large Language Models",
            "content": "Large language models (LLMs) are neural networks trained on vast text corpora to understand and generate human language.",
            "category": "AI Fundamentals",
            "created_at": datetime(2024, 1, 10),
        },
        {
            "title": "Retrieval-Augmented Generation",
            "content": "RAG combines a retrieval system with a generative model, allowing the model to ground its responses in external knowledge sources.",
            "category": "AI Techniques",
            "created_at": datetime(2024, 2, 15),
        },
        {
            "title": "Vector Databases Explained",
            "content": "Vector databases store high-dimensional embeddings and enable fast approximate nearest-neighbor search for semantic retrieval tasks.",
            "category": "Infrastructure",
            "created_at": datetime(2024, 3, 5),
        },
        {
            "title": "Prompt Engineering Best Practices",
            "content": "Effective prompt engineering involves clear instructions, few-shot examples, and structured output formatting to guide model behavior.",
            "category": "AI Techniques",
            "created_at": datetime(2024, 4, 20),
        },
        {
            "title": "AI Agents and Tool Use",
            "content": "AI agents autonomously plan and execute multi-step tasks by calling external tools such as search engines, databases, and APIs.",
            "category": "AI Agents",
            "created_at": datetime(2024, 5, 30),
        },
    ]

    with Session(engine) as session:
        session.execute(documents_table.insert(), sample_rows)
        session.commit()

    print(f"SQLite seeding complete — {len(sample_rows)} rows inserted into '{SQLITE_DB_PATH}'")


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
    seed_sqlite()
