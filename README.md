title: Research
emoji: 👁
colorFrom: blue
colorTo: red
sdk: docker
pinned: false

# Research- RAG Chat App (LangChain + Chroma + Groq)

This project builds a context-grounded LLM chat application using your
`research_context.json` file.

It does the following:

1. Loads and parses `research_context.json`.
2. Chunks text content with a recursive text splitter.
3. Embeds chunks with a lightweight Hugging Face model.
4. Stores vectors in a local Chroma database.
5. Runs a chat loop with Groq LLM and LangChain memory support.

## Project Files

- `rag_app/ingest.py`: Parse JSON, chunk, embed, and index in Chroma.
- `rag_app/chat.py`: Conversational RAG app with chat memory.
- `.env.example`: Environment variable template.
- `requirements.txt`: Python dependencies.

## 1) Install Dependencies

```powershell
pip install -r requirements.txt
```

## 2) Configure Environment

Create a `.env` file in project root from `.env.example` and set your Groq key.

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

## 3) Build the Vector Store (Chunk + Embed + Store)

```powershell
python rag_app/ingest.py --input research_context.json --persist-dir chroma_db
```

Default embedding model:

- `sentence-transformers/all-MiniLM-L6-v2` (lightweight and fast)

## 4) Start Chat

```powershell
python rag_app/chat.py --persist-dir chroma_db
```

Optional scaling-friendly persistence flags:

```powershell
python rag_app/chat.py --persist-dir chroma_db --session-id team_a --memory-db chat_memory.db --reference-db chat_reference.db
```

Chat commands:

- `/sources` toggle showing retrieved sources
- `/exit` quit

## 5) Start Streamlit UI

Run the modern web UI with cyberpunk dark styling:

```powershell
streamlit run rag_app/streamlit_app.py
```

The UI provides:

- Persistent conversational memory (`chat_memory.db`)
- Turn-level reference logging (`chat_reference.db`)
- Source visibility toggle
- Adjustable model and retrieval settings from the sidebar

## Notes

- The assistant is instructed to answer only from retrieved context.
- If relevant context is missing, it should say it does not know from provided documents.
- Chat memory is managed using `RunnableWithMessageHistory` + `SQLChatMessageHistory`.
- Memory is persisted in SQLite (`--memory-db`) so sessions survive restarts.
- Each turn is also logged in a simple SQLite reference table (`--reference-db`) with source metadata.
