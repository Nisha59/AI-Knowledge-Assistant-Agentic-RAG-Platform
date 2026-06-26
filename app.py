import uvicorn
import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents.main_agent import MainAgent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str


@app.get("/health")
def health():
    return {"status": "ok", "model": "gpt-4o"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        answer, _ = MainAgent().run(request.question)
        return QueryResponse(question=request.question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


EMPTY_STATE = "*Ask a question to see results.*"
SEARCHING_STATE = "*Searching...*"


def format_semantic(results: list) -> str:
    if not results:
        return "*No knowledge base results retrieved.*"
    parts = []
    for i, r in enumerate(results, 1):
        source = r.get("source", "unknown")
        content = r.get("content", "")[:300]
        parts.append(f"**Result {i}** &nbsp;·&nbsp; `{source}`\n\n{content}…")
    return "\n\n---\n\n".join(parts)


def format_sql(results: list) -> str:
    if not results:
        return "*No database results retrieved.*"
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"**Row {i}**\n\n{r.get('content', '')}")
    return "\n\n---\n\n".join(parts)


def format_web(results: list) -> str:
    if not results:
        return "*No web results retrieved.*"
    parts = []
    for i, r in enumerate(results, 1):
        url = r.get("source", "")
        content = r.get("content", "")[:300]
        parts.append(f"**Result {i}**\n\n🔗 [{url}]({url})\n\n{content}…")
    return "\n\n---\n\n".join(parts)


def chat(message, history):
    history.append({"role": "user", "content": message})
    yield "", history, SEARCHING_STATE, SEARCHING_STATE, SEARCHING_STATE

    answer, sources = MainAgent().run(message)
    history.append({"role": "assistant", "content": answer})
    yield (
        "",
        history,
        format_semantic(sources["semantic"]),
        format_sql(sources["sql"]),
        format_web(sources["web"]),
    )


def clear():
    return [], EMPTY_STATE, EMPTY_STATE, EMPTY_STATE


with gr.Blocks(title="AI Knowledge Assistant", theme=gr.themes.Soft()) as ui:
    gr.Markdown("# AI Knowledge Assistant")
    gr.Markdown("Ask questions — the agent searches your knowledge base, database, and the web.")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages", height=520, show_label=False)
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask a question...",
                    show_label=False,
                    scale=5,
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear Chat", size="sm")

        with gr.Column(scale=2):
            gr.Markdown("## Retrieved Sources")

            with gr.Accordion("📚 Knowledge Base", open=True):
                semantic_display = gr.Markdown(EMPTY_STATE)

            with gr.Accordion("🗄️ Database", open=False):
                sql_display = gr.Markdown(EMPTY_STATE)

            with gr.Accordion("🌐 Web Search", open=False):
                web_display = gr.Markdown(EMPTY_STATE)

    outputs = [msg_box, chatbot, semantic_display, sql_display, web_display]

    submit_btn.click(chat, inputs=[msg_box, chatbot], outputs=outputs)
    msg_box.submit(chat, inputs=[msg_box, chatbot], outputs=outputs)
    clear_btn.click(clear, outputs=[chatbot, semantic_display, sql_display, web_display])


app = gr.mount_gradio_app(app, ui, path="/")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
