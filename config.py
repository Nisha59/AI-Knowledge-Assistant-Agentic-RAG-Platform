import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./vector_db")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./knowledge.db")
