from __future__ import annotations

import os
import re
from typing import List

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

INDEX_DIR = os.getenv("INDEX_DIR", "outputs/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5-mini-2025-08-07")
TOP_K = int(os.getenv("TOP_K", "5"))

CITATION = re.compile(r"\[[^:\]]+:p\d+\]")


def load_db() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
        persist_directory=INDEX_DIR,
    )


def format_docs(docs: List[Document]) -> str:
    lines: List[str] = []
    for d in docs:
        doc_id = d.metadata.get("doc_id", "?")
        page_1 = d.metadata.get("page_1", "?")
        text = (d.page_content or "").replace("\n", " ")
        lines.append(f"[{doc_id}:p{page_1}] {text}")
    return "\n".join(lines)


def enforce_citation(answer: str) -> str:
    text = (answer or "").strip()
    if CITATION.search(text):
        return text
    return "I don't know based on the provided documents. (no citation found)"


def build_chain():
    db = load_db()
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a document-grounded assistant.\n"
                "Use ONLY the provided context.\n"
                "Cite sources inline as [doc_id:pX].\n"
                "If the context is insufficient, say you don't know.",
            ),
            ("human", "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:"),
        ]
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | ChatOpenAI(model=CHAT_MODEL)
        | StrOutputParser()
    )
    return chain, retriever


def main():
    chain, retriever = build_chain()

    while True:
        q = input("\nQuestion (exit to quit): ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        ans = enforce_citation(chain.invoke(q))
        docs = retriever.invoke(q)

        print("\n--- Answer ---")
        print(ans)

        print("\n--- Retrieved ---")
        for d in docs:
            print(f"[{d.metadata.get('doc_id')}:p{d.metadata.get('page_1')}] {d.metadata.get('source')}")


if __name__ == "__main__":
    main()
