import json
import openai
import config
from agents.retrieval_agent import RetrievalAgent

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieval_agent",
            "description": (
                "Retrieves relevant context from the internal knowledge base "
                "(semantic search), SQL database, and web search for a given query. "
                "Always call this before answering any factual question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to retrieve context for",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to a powerful knowledge retrieval "
    "system. When the user asks a question, always call the retrieval_agent tool "
    "first to gather relevant context. Then answer the question based on the "
    "retrieved context. Be accurate, concise, and cite sources where possible."
)


class MainAgent:
    def __init__(self):
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

    def run(self, question: str) -> tuple[str, dict]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        all_sources = {"semantic": [], "sql": [], "web": []}

        for _ in range(5):
            response = self.client.chat.completions.create(
                model="gpt-4o",
                tools=TOOLS,
                messages=messages,
            )

            message = response.choices[0].message

            if message.tool_calls is None:
                return message.content or "", all_sources

            messages.append(message)

            for tool_call in message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                query = args["query"]
                context, sources = RetrievalAgent().run(query)
                for key in all_sources:
                    all_sources[key].extend(sources[key])
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": context,
                })

        return response.choices[0].message.content or "", all_sources
