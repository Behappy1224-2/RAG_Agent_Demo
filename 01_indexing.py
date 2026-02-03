from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "data/docs")
INDEX_DIR = os.getenv("INDEX_DIR", "outputs/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md"}


def iter_files(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {root.resolve()}")
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    if not files:
        raise RuntimeError(f"No supported files under {root.resolve()} (pdf/docx/txt/md).")
    return sorted(files)


def load_one(path: Path) -> List[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    elif ext == ".docx":
        docs = Docx2txtLoader(str(path)).load()
    else:
        docs = TextLoader(str(path), encoding="utf-8").load()

    doc_id = path.stem
    for d in docs:
        md = dict(d.metadata or {})
        md["doc_id"] = doc_id
        md["source"] = str(path)
        md["file_ext"] = ext

        # PDF page -> 1-index for citations
        if ext == ".pdf":
            page0 = md.get("page", md.get("page_number", 0))
            md["page_1"] = int(page0) + 1 if isinstance(page0, int) else 1
        else:
            md["page_1"] = 1

        d.metadata = md
    return docs


def stable_chunk_id(doc: Document) -> str:
    """
    Stable ID based on (doc_id, page_1, chunk text).
    This is resilient to file insertions/reordering.
    """
    doc_id = str(doc.metadata.get("doc_id", "unknown"))
    page_1 = str(doc.metadata.get("page_1", "1"))
    text = (doc.page_content or "").strip()
    raw = f"{doc_id}|p{page_1}|{text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:24]


def split_and_id(docs: List[Document]) -> Tuple[List[Document], List[str]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    ids: List[str] = []
    for c in chunks:
        cid = stable_chunk_id(c)
        c.metadata["chunk_id"] = cid
        ids.append(cid)
    return chunks, ids


def build_index(reset: bool) -> int:
    data_root = Path(DATA_DIR)
    index_root = Path(INDEX_DIR)

    if reset and index_root.exists():
        shutil.rmtree(index_root)

    files = iter_files(data_root)

    all_docs: List[Document] = []
    for f in files:
        all_docs.extend(load_one(f))

    chunks, ids = split_and_id(all_docs)

    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
        persist_directory=INDEX_DIR,
    )

    # avoid duplicate IDs if not reset (simple demo approach)
    try:
        existing = set(db.get(include=[])["ids"])
    except Exception:
        existing = set()

    new_chunks = []
    new_ids = []
    for c, cid in zip(chunks, ids):
        if cid not in existing:
            new_chunks.append(c)
            new_ids.append(cid)

    if new_chunks:
        db.add_documents(new_chunks, ids=new_ids)

    print(f"✅ Indexed {len(new_chunks)} new chunks (total scanned: {len(chunks)}) from {len(files)} files")
    print(f"   persist_dir: {Path(INDEX_DIR).resolve()}")
    print(f"   collection : {COLLECTION_NAME}")
    return len(new_chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="Delete existing index and rebuild.")
    args = ap.parse_args()
    build_index(reset=args.reset)


if __name__ == "__main__":
    main()
