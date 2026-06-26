import openai
import chromadb
from sqlalchemy import create_engine, text
from tavily import TavilyClient
import config


def retrieve_semantic(query: str) -> list[dict]:
    try:
        client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.embeddings.create(input=query, model="text-embedding-3-large")
        query_embedding = response.data[0].embedding

        chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        collection = chroma_client.get_collection("langchain")
        results = collection.query(query_embeddings=[query_embedding], n_results=5)

        docs = []
        for i, doc in enumerate(results["documents"][0]):
            source = results["metadatas"][0][i].get("filename", "unknown")
            docs.append({"source": source, "content": doc})
        return docs
    except Exception:
        return []


def retrieve_sql(query: str) -> list[dict]:
    try:
        client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Convert the user's natural language query into a SQL SELECT "
                        "statement for a SQLite table called 'documents' with columns: "
                        "id, title, content, category, created_at. "
                        "Return only the SQL statement, no explanation."
                    ),
                },
                {"role": "user", "content": query},
            ],
        )

        sql = response.choices[0].message.content.strip()
        if sql.startswith("```"):
            lines = sql.splitlines()
            sql = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        engine = create_engine(f"sqlite:///{config.SQLITE_DB_PATH}")
        with engine.connect() as conn:
            rows = conn.execute(text(sql)).fetchall()

        return [{"source": "sql", "content": str(row)} for row in rows]
    except Exception:
        return []


def retrieve_web(query: str) -> list[dict]:
    try:
        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        response = client.search(query, max_results=5)
        return [{"source": r["url"], "content": r["content"]} for r in response["results"]]
    except Exception:
        return []
