import json
import openai
import config
from tools.retrieval_tools import retrieve_semantic, retrieve_sql, retrieve_web

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_semantic",
            "description": "Search the internal knowledge base using semantic similarity",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_sql",
            "description": "Query structured data from the SQL database using natural language",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_web",
            "description": "Search the web for current or external information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_MAP = {
    "retrieve_semantic": retrieve_semantic,
    "retrieve_sql": retrieve_sql,
    "retrieve_web": retrieve_web,
}

SOURCE_CATEGORY = {
    "retrieve_semantic": "semantic",
    "retrieve_sql": "sql",
    "retrieve_web": "web",
}

SYSTEM_PROMPT = (
    "You are a retrieval agent. Your job is to find relevant context for a given query "
    "by calling the most appropriate tool(s). Use this logic to decide:\n\n"
    "- retrieve_semantic: for questions about company-specific topics — people, products, "
    "policies, contracts, awards, internal procedures, or anything likely in company documents.\n"
    "- retrieve_sql: for structured data queries — counts, lists, categories, or records "
    "from the company database.\n"
    "- retrieve_web: for general knowledge, current events, sports, news, or anything "
    "that is clearly NOT company-internal information.\n\n"
    "Only call the tools that are relevant to the query. Do not call all three by default. "
    "If a question spans both internal and external knowledge, call multiple tools. "
    "After collecting results, return a single consolidated summary of all retrieved context."
)


class RetrievalAgent:
    def __init__(self):
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

    def run(self, query: str) -> tuple[str, dict]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        sources = {"semantic": [], "sql": [], "web": []}

        for _ in range(5):
            response = self.client.chat.completions.create(
                model="gpt-4o",
                tools=TOOLS,
                messages=messages,
            )

            message = response.choices[0].message

            if message.tool_calls is None:
                return message.content or "", sources

            messages.append(message)

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                result = TOOL_MAP[name](**args)
                category = SOURCE_CATEGORY.get(name, "semantic")
                sources[category].extend(result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })

        return response.choices[0].message.content or "", sources
