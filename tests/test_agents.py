import json
from unittest.mock import MagicMock, patch
from tools.retrieval_tools import retrieve_semantic, retrieve_sql, retrieve_web
from agents.retrieval_agent import RetrievalAgent
from agents.main_agent import MainAgent


# ---------------------------------------------------------------------------
# helpers to build mock OpenAI response objects
# ---------------------------------------------------------------------------

def _make_tool_call(name, arguments: dict, call_id="call_1"):
    tool_call = MagicMock()
    tool_call.id = call_id
    tool_call.function.name = name
    tool_call.function.arguments = json.dumps(arguments)
    return tool_call


def _make_tool_response(name, arguments: dict, call_id="call_1"):
    msg = MagicMock()
    msg.tool_calls = [_make_tool_call(name, arguments, call_id)]
    msg.content = None
    resp = MagicMock()
    resp.choices[0].message = msg
    return resp


def _make_text_response(text: str):
    msg = MagicMock()
    msg.tool_calls = None
    msg.content = text
    resp = MagicMock()
    resp.choices[0].message = msg
    return resp


# ---------------------------------------------------------------------------
# test_retrieve_semantic
# ---------------------------------------------------------------------------

@patch("tools.retrieval_tools.chromadb.PersistentClient")
@patch("tools.retrieval_tools.openai.OpenAI")
def test_retrieve_semantic(mock_openai_cls, mock_chroma_cls):
    # mock embedding
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.embeddings.create.return_value.data[0].embedding = [0.1] * 10

    # mock chroma collection query
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["AI is transforming industries."]],
        "metadatas": [[{"filename": "ai_intro.md"}]],
    }
    mock_chroma_cls.return_value.get_collection.return_value = mock_collection

    result = retrieve_semantic("artificial intelligence")

    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# test_retrieve_sql
# ---------------------------------------------------------------------------

@patch("tools.retrieval_tools.create_engine")
@patch("tools.retrieval_tools.openai.OpenAI")
def test_retrieve_sql(mock_openai_cls, mock_engine_cls):
    # mock LLM returning a SQL string
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value.choices[
        0
    ].message.content = "SELECT * FROM documents LIMIT 3"

    # mock sqlalchemy engine returning fake rows
    fake_row = ("1", "AI Doc", "Some content", "AI", "2024-01-01")
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = [fake_row]
    mock_engine_cls.return_value.connect.return_value.__enter__ = MagicMock(
        return_value=mock_conn
    )
    mock_engine_cls.return_value.connect.return_value.__exit__ = MagicMock(
        return_value=False
    )

    result = retrieve_sql("show me all AI documents")

    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# test_retrieve_web
# ---------------------------------------------------------------------------

@patch("tools.retrieval_tools.DDGS")
def test_retrieve_web(mock_ddgs_cls):
    mock_ddgs_cls.return_value.text.return_value = [
        {"href": "http://test.com", "body": "test content"}
    ]

    result = retrieve_web("what is RAG")

    assert isinstance(result, list)
    assert len(result) == 1
    assert "source" in result[0]
    assert "content" in result[0]


# ---------------------------------------------------------------------------
# test_retrieval_agent
# ---------------------------------------------------------------------------

@patch("agents.retrieval_agent.retrieve_semantic", return_value=[{"source": "doc.md", "content": "AI info"}])
@patch("agents.retrieval_agent.openai.OpenAI")
def test_retrieval_agent(mock_openai_cls, mock_retrieve):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.side_effect = [
        _make_tool_response("retrieve_semantic", {"query": "test query"}),
        _make_text_response("Here is the context"),
    ]

    result = RetrievalAgent().run("test query")

    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# test_main_agent
# ---------------------------------------------------------------------------

@patch("agents.main_agent.RetrievalAgent")
@patch("agents.main_agent.openai.OpenAI")
def test_main_agent(mock_openai_cls, mock_retrieval_agent_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.side_effect = [
        _make_tool_response("retrieval_agent", {"query": "What do you know?"}),
        _make_text_response("Here is your answer"),
    ]

    mock_retrieval_agent_cls.return_value.run.return_value = "some context"

    result = MainAgent().run("What do you know?")

    assert isinstance(result, str)
    assert len(result) > 0
