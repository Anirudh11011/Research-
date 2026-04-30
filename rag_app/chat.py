import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with your indexed research context")
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="chroma_db",
        help="Directory containing persisted Chroma DB",
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
        "--model",
        type=str,
        default=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        help="Groq model name",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of retrieved chunks",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="default_session",
        help="Chat session id for memory",
    )
    parser.add_argument(
        "--memory-db",
        type=str,
        default="chat_memory.db",
        help="SQLite file used by SQLChatMessageHistory",
    )
    parser.add_argument(
        "--reference-db",
        type=str,
        default="chat_reference.db",
        help="SQLite file to store Q/A references and source metadata",
    )
    return parser.parse_args()


def build_chain(
    persist_dir: str,
    collection: str,
    embedding_model: str,
    model_name: str,
    temperature: float,
    k: int,
    memory_db_path: str,
) -> RunnableWithMessageHistory:
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=collection,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = ChatGroq(model=model_name, temperature=temperature)

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, rewrite the latest "
        "question so it can be understood without chat history. Do not answer it."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()

    qa_system_prompt = (
        "You are a research assistant. Answer only from the provided context. "
        "If the answer is not present in the context, say: "
        "'I do not have enough information in the provided documents.' "
        "Keep answers clear and concise.\n\n"
        "Context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    qa_chain = qa_prompt | llm | StrOutputParser()

    def format_docs(docs: list) -> str:
        return "\n\n".join([doc.page_content for doc in docs])

    def run_rag(payload: dict) -> dict:
        user_input = payload["input"]
        chat_history = payload.get("chat_history", [])

        if chat_history:
            standalone_question = contextualize_chain.invoke(
                {"input": user_input, "chat_history": chat_history}
            )
        else:
            standalone_question = user_input

        docs = retriever.invoke(standalone_question)
        answer = qa_chain.invoke(
            {
                "input": user_input,
                "chat_history": chat_history,
                "context": format_docs(docs),
            }
        )
        return {"answer": answer, "context": docs}

    rag_chain = RunnableLambda(run_rag)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=f"sqlite:///{memory_db_path}",
            table_name="chat_message_history",
        )

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain


def extract_unique_sources(response: dict) -> List[Tuple[str, object]]:
    docs = response.get("context", [])
    seen = set()
    sources: List[Tuple[str, object]] = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append(key)

    return sources


def format_sources(response: dict) -> str:
    sources = extract_unique_sources(response)
    if not sources:
        return "No sources returned."

    lines = []
    for source, page in sources:
        lines.append(f"- {source} (page: {page})")
    return "\n".join(lines)


def init_reference_db(reference_db_path: str) -> None:
    with sqlite3.connect(reference_db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_reference (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                session_id TEXT NOT NULL,
                user_query TEXT NOT NULL,
                assistant_answer TEXT NOT NULL,
                sources_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def log_turn_to_reference_db(
    reference_db_path: str,
    session_id: str,
    user_query: str,
    assistant_answer: str,
    sources: List[Tuple[str, object]],
) -> None:
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    sources_payload = [{"source": source, "page": page} for source, page in sources]

    with sqlite3.connect(reference_db_path) as conn:
        conn.execute(
            """
            INSERT INTO conversation_reference (
                timestamp_utc,
                session_id,
                user_query,
                assistant_answer,
                sources_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                timestamp_utc,
                session_id,
                user_query,
                assistant_answer,
                json.dumps(sources_payload, ensure_ascii=True),
            ),
        )
        conn.commit()


def print_reference_help(reference_db_path: str) -> None:
    print("\nReference DB saved at:", reference_db_path)
    print("Example SQL query:")
    print(
        "SELECT id, timestamp_utc, session_id, user_query FROM conversation_reference ORDER BY id DESC LIMIT 10;"
    )


def main() -> None:
    load_dotenv()
    args = parse_args()

    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError(
            "GROQ_API_KEY is missing. Add it in your environment or .env file."
        )

    init_reference_db(args.reference_db)

    chain = build_chain(
        persist_dir=args.persist_dir,
        collection=args.collection,
        embedding_model=args.embedding_model,
        model_name=args.model,
        temperature=args.temperature,
        k=args.k,
        memory_db_path=args.memory_db,
    )

    print("RAG chat started. Type '/exit' to quit. Type '/sources' to toggle source display.")
    print_reference_help(args.reference_db)

    show_sources = False
    while True:
        query = input("\nYou: ").strip()

        if not query:
            continue
        if query.lower() == "/exit":
            print("Exiting chat.")
            break
        if query.lower() == "/sources":
            show_sources = not show_sources
            print(f"Show sources: {show_sources}")
            continue

        response = chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": args.session_id}},
        )
        answer = response.get("answer", "")
        sources = extract_unique_sources(response)

        log_turn_to_reference_db(
            reference_db_path=args.reference_db,
            session_id=args.session_id,
            user_query=query,
            assistant_answer=answer,
            sources=sources,
        )

        print(f"\nAssistant: {answer}")

        if show_sources:
            print("\nSources:")
            print(format_sources(response))


if __name__ == "__main__":
    main()
