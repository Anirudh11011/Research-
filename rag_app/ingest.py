import argparse
import json
import os
import shutil
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest research_context.json into Chroma DB"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="research_context.json",
        help="Path to research_context.json",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="chroma_db",
        help="Directory where Chroma data is stored",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="research_context",
        help="Chroma collection name",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face embedding model",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for text splitting",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Chunk overlap for text splitting",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing Chroma directory before ingesting",
    )
    return parser.parse_args()


def _clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join([line for line in lines if line])


def _extract_page_text(page: dict[str, Any]) -> str:
    parts: list[str] = []

    if isinstance(page.get("text"), str):
        parts.append(page["text"])

    for table in page.get("tables", []) or []:
        markdown = table.get("markdown")
        if isinstance(markdown, str) and markdown.strip():
            parts.append(markdown)

    return _clean_text("\n\n".join(parts))


def _fallback_collect_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool)):
        return str(obj)
    if isinstance(obj, list):
        return "\n".join(
            [part for part in (_fallback_collect_text(item) for item in obj) if part]
        )
    if isinstance(obj, dict):
        preferred_keys = ["text", "markdown", "content", "title", "abstract"]
        parts: list[str] = []

        for key in preferred_keys:
            if key in obj:
                val = _fallback_collect_text(obj[key])
                if val:
                    parts.append(val)

        for key, val in obj.items():
            if key in preferred_keys:
                continue
            if isinstance(val, (dict, list)):
                nested = _fallback_collect_text(val)
                if nested:
                    parts.append(nested)

        return "\n".join(parts)

    return ""


def load_documents_from_json(input_path: str) -> list[Document]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents: list[Document] = []

    for item in data.get("documents", []):
        file_name = item.get("file_name", "unknown_source")
        drive_url = item.get("drive_url", "")
        content = item.get("content", {})
        pages = content.get("pages", []) if isinstance(content, dict) else []

        if isinstance(pages, list) and pages:
            for page in pages:
                page_number = page.get("page_number")
                page_text = _extract_page_text(page)
                if not page_text:
                    continue

                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": file_name,
                            "page": page_number,
                            "drive_url": drive_url,
                        },
                    )
                )
        else:
            raw_text = _clean_text(_fallback_collect_text(content))
            if raw_text:
                documents.append(
                    Document(
                        page_content=raw_text,
                        metadata={
                            "source": file_name,
                            "page": None,
                            "drive_url": drive_url,
                        },
                    )
                )

    return documents


def main() -> None:
    args = parse_args()

    if args.reset and os.path.exists(args.persist_dir):
        shutil.rmtree(args.persist_dir)

    docs = load_documents_from_json(args.input)
    if not docs:
        raise ValueError("No text documents found in input JSON.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=args.collection,
        persist_directory=args.persist_dir,
        embedding_function=embeddings,
    )
    vectorstore.add_documents(chunks)

    print(f"Loaded source pages: {len(docs)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Persisted Chroma DB at: {args.persist_dir}")
    print(f"Collection name: {args.collection}")
    print(f"Embedding model: {args.embedding_model}")


if __name__ == "__main__":
    main()
