from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple, Any

from dotenv import load_dotenv

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool

from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

load_dotenv()

INDEX_DIR = os.getenv("INDEX_DIR", "outputs/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5-mini-2025-08-07")
TOP_K = int(os.getenv("TOP_K", "5"))

THREAD_ID = os.getenv("THREAD_ID", "1")
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "14"))  # max steps per user turn (model/tool hops)
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "10"))        # bound short-term memory size (messages kept)


CITATION = re.compile(r"\[[^:\]]+:p\d+\]")


def vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
        persist_directory=INDEX_DIR,
    )


def format_docs(docs: List[Document]) -> str:
    if not docs:
        return "NO_EVIDENCE_FOUND"
    lines: List[str] = []
    for d in docs:
        doc_id = d.metadata.get("doc_id", "?")
        page_1 = d.metadata.get("page_1", "?")
        text = (d.page_content or "").replace("\n", " ")
        lines.append(f"[{doc_id}:p{page_1}] {text}")
    return "\n".join(lines)


def extract_sources(messages: List[Any]) -> List[dict]:
    sources: List[dict] = []
    for m in messages:
        artifact = getattr(m, "artifact", None)
        if isinstance(artifact, list) and all(isinstance(d, Document) for d in artifact):
            for d in artifact:
                sources.append(
                    {
                        "doc_id": d.metadata.get("doc_id"),
                        "page_1": d.metadata.get("page_1"),
                        "source": d.metadata.get("source"),
                    }
                )
    uniq = {}
    for s in sources:
        uniq[(s.get("doc_id"), s.get("page_1"), s.get("source"))] = s
    return list(uniq.values())


def get_last_content(messages: List[Any]) -> str:
    last = messages[-1]
    if isinstance(last, dict):
        return str(last.get("content", ""))
    return str(getattr(last, "content", ""))


@before_model
def trim_short_term_memory(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state.get("messages", [])
    if len(messages) <= MAX_MESSAGES:
        return None

    kept = messages[-MAX_MESSAGES:]

    # Remove all previous messages, then add back only the trimmed window.
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *kept,
        ]
    }


def build_agent(k: int = TOP_K):
    retriever = vectorstore().as_retriever(search_kwargs={"k": k})

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str, doc_id: Optional[str] = None) -> Tuple[str, List[Document]]:
        """Retrieve relevant chunks from the local vector index. Optionally filter by doc_id."""
        docs = retriever.invoke(query)
        if doc_id:
            docs = [d for d in docs if d.metadata.get("doc_id") == doc_id]
        return format_docs(docs), docs

    system_prompt = (
        "You are an agent with two modes:\n"
        "MODE A (Document-grounded / RAG): If the question is about the provided documents (paper content, "
        "claims, numbers, company info in docs), call retrieve_context. Use ONLY retrieved context and cite "
        "inline exactly like [doc_id:pX]. If insufficient, say so.\n"
        "MODE B (General): If the question is general knowledge (definitions like ML/LLM), answer directly "
        "WITHOUT calling tools. Label it as GENERAL and do not fabricate citations.\n"
        "You may call retrieve_context multiple times if needed, but stay within the recursion limit.\n"
    )

    agent = create_agent(
        ChatOpenAI(model=CHAT_MODEL),
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),              # short-term memory
        middleware=[trim_short_term_memory],       # bound memory size
    ).with_config({"recursion_limit": RECURSION_LIMIT})      

    return agent


def main():
    agent = build_agent()

    # # --- SAVE GRAPH AS PNG---
    # try:
    #     png_bytes = agent.get_graph(xray=True).draw_mermaid_png()
    #     out_path = "agent_graph.png"
    #     with open(out_path, "wb") as f:
    #         f.write(png_bytes)
    #     print(f"Saved graph to: {out_path}")
    # except Exception as e:
    #     print("Graph PNG render failed:", repr(e))
    #     print("Mermaid text fallback:\n")
    #     print(agent.get_graph().draw_mermaid())

    while True:
        q = input("\nQuestion (exit to quit): ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        res = agent.invoke(
            {"messages": [{"role": "user", "content": q}]},
            {"configurable": {"thread_id": THREAD_ID}},
        )

        messages = res.get("messages", [])
        final = get_last_content(messages).strip()

        srcs = extract_sources(messages)
        if srcs and CITATION.search(final) is None:
            final = "I don't know based on the provided documents. (no citation found)"

        print("\n--- Answer ---")
        print(final)

        if srcs:
            print("\n--- Sources (from tool artifacts) ---")
            for s in srcs:
                print(f"[{s['doc_id']}:p{s['page_1']}] {s['source']}")


if __name__ == "__main__":
    main()
