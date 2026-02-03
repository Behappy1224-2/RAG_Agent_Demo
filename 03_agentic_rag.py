from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple, Any

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver  

load_dotenv()

INDEX_DIR = os.getenv("INDEX_DIR", "outputs/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5-mini-2025-08-07")
TOP_K = int(os.getenv("TOP_K", "5"))

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
    """
            You are a professional document analysis assistant, primarily responsible for parsing and analyzing document content.
            Please adhere to the following operational guidelines to ensure the accuracy of your responses:

            Tool Priority:
            When the user asks about specific technical details, experimental data, comparison targets, or algorithmic workflows, you must first call the retrieve_context tool.
            Do not rely on general knowledge to answer; responses must be based strictly on the retrieved content.

            Structured Responses:
            If the answer involves multiple contributions, steps, or comparison targets, present the information using headings and lists.
            For experimental data, list the results in a concise and clear format whenever possible.

            Honesty and Boundaries:
            If the retrieved content is insufficient to answer the question, explicitly state: “The provided document does not mention the relevant details,” and do not fabricate information.
            If the user’s question is ambiguous, ask for clarification based on the document context (e.g., “Are you referring to which experimental phase in the paper?”).

            Citation Standards:
            When answering, reference the source whenever possible (e.g., “According to the document’s description of the experimental setup…”).

            Language Style:
            Respond in English, while retaining necessary technical terms .
            
            Current Affairs Questions:
            If the user asks about current affairs, you may also provide an answer (e.g., “Who is the current President of the United States?”).
        """
    )

    agent = create_agent(
        ChatOpenAI(model=CHAT_MODEL),
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),  
    )
    return agent


def main():
    agent = build_agent()

    while True:
        q = input("\nQuestion (exit to quit): ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        res = agent.invoke({"messages": [{"role": "user", "content": q}]},
                           {"configurable": {"thread_id": "1"}})
        messages = res.get("messages", [])

        final = get_last_content(messages).strip()

        print("\n--- Answer ---")
        print(final)

        srcs = extract_sources(messages)
        if srcs:
            print("\n--- Sources (from tool artifacts) ---")
            for s in srcs:
                print(f"[{s['doc_id']}:p{s['page_1']}] {s['source']}")


if __name__ == "__main__":
    main()
