import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from chat import build_chain, extract_unique_sources, init_reference_db, log_turn_to_reference_db


def inject_custom_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Space+Grotesk:wght@400;500;700&display=swap');

:root {
    --deep-forest: #0f2a1e;
    --midnight-purple: #1b1333;
    --electric-teal: #00d6c3;
    --slate-gray: #52606d;
    --bg-0: #070b12;
    --bg-1: #0d121b;
    --bg-2: #131a23;
    --jewel-emerald: #1de39a;
    --jewel-cyan: #20e6ff;
    --jewel-magenta: #b26bff;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 12% 8%, rgba(0, 214, 195, 0.15), transparent 35%),
        radial-gradient(circle at 88% 12%, rgba(178, 107, 255, 0.12), transparent 30%),
        radial-gradient(circle at 30% 92%, rgba(29, 227, 154, 0.10), transparent 35%),
        linear-gradient(145deg, var(--bg-0) 0%, var(--bg-1) 55%, var(--bg-2) 100%);
    color: #ecf2f6;
}

h1, h2, h3 {
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 0.06em;
    color: #ecf9ff;
}

[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15, 42, 30, 0.72) 0%, rgba(27, 19, 51, 0.72) 100%);
    border-right: 1px solid rgba(32, 230, 255, 0.22);
}

[data-testid="stSidebar"] * {
    color: #d8e4ec;
}

[data-testid="stChatMessage"] {
    border: 1px solid rgba(82, 96, 109, 0.32);
    border-radius: 14px;
    padding: 0.45rem 0.65rem;
    backdrop-filter: blur(5px);
    animation: riseIn 0.28s ease-out;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(15, 42, 30, 0.68), rgba(0, 214, 195, 0.14));
    border-color: rgba(29, 227, 154, 0.38);
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: linear-gradient(135deg, rgba(27, 19, 51, 0.72), rgba(32, 230, 255, 0.12));
    border-color: rgba(178, 107, 255, 0.34);
}

.stChatInputContainer {
    background: linear-gradient(90deg, rgba(19, 26, 35, 0.98), rgba(13, 18, 27, 0.98));
    border-top: 1px solid rgba(32, 230, 255, 0.24);
}

[data-testid="stChatInput"] {
    border: 1px solid rgba(82, 96, 109, 0.42);
}

.stButton > button {
    background: linear-gradient(135deg, rgba(0, 214, 195, 0.18), rgba(178, 107, 255, 0.22));
    color: #ecf2f6;
    border: 1px solid rgba(32, 230, 255, 0.3);
    border-radius: 10px;
}

.stButton > button:hover {
    border-color: rgba(29, 227, 154, 0.64);
    box-shadow: 0 0 0 1px rgba(29, 227, 154, 0.2), 0 0 16px rgba(0, 214, 195, 0.2);
}

.app-caption {
    color: #a9bac7;
    font-size: 0.95rem;
    margin-top: -0.4rem;
    margin-bottom: 0.7rem;
}

@keyframes riseIn {
    from {
        transform: translateY(8px);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}
</style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_chain(
    persist_dir: str,
    collection: str,
    embedding_model: str,
    model_name: str,
    temperature: float,
    k: int,
    memory_db: str,
):
    return build_chain(
        persist_dir=persist_dir,
        collection=collection,
        embedding_model=embedding_model,
        model_name=model_name,
        temperature=temperature,
        k=k,
        memory_db_path=memory_db,
    )


def format_sources_md(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "No sources returned."
    lines = []
    for item in sources:
        lines.append(f"- {item.get('source', 'unknown')} (page: {item.get('page')})")
    return "\n".join(lines)


def main() -> None:
    load_dotenv()

    st.set_page_config(
        page_title="SEAR Lab AI Chat Assistant",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    st.title("SEAR Lab AI Assistant")
    st.markdown(
        '<p class="app-caption">Hi! I am SEAR Lab AI Chat Assistant. Ask me any question about our Lab research work </p>',
        unsafe_allow_html=True,
    )

    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is missing. Add it in your environment or .env file.")
        st.stop()

    with st.sidebar:
        st.header("Configuration")

        persist_dir = st.text_input("Persist directory", value="chroma_db")
        collection = st.text_input("Collection name", value="research_context")
        embedding_model = st.text_input(
            "Embedding model", value="sentence-transformers/all-MiniLM-L6-v2"
        )
        model_name = st.text_input(
            "Groq model", value=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        )
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0)
        k = st.slider("Top-k retrieval", min_value=1, max_value=12, value=4)

        st.divider()
        session_id = st.text_input("Session ID", value="streamlit_session")
        memory_db = st.text_input("Memory DB", value="chat_memory.db")
        reference_db = st.text_input("Reference DB", value="chat_reference.db")
        show_sources = st.toggle("Show sources", value=True)

        if st.button("Clear current UI chat"):
            st.session_state.messages = []
            st.rerun()

    init_reference_db(reference_db)

    chain = get_chain(
        persist_dir=persist_dir,
        collection=collection,
        embedding_model=embedding_model,
        model_name=model_name,
        temperature=temperature,
        k=k,
        memory_db=memory_db,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if show_sources and message["role"] == "assistant":
                sources = message.get("sources", [])
                if sources:
                    with st.expander("Sources"):
                        st.markdown(format_sources_md(sources))

    prompt = st.chat_input("Ask something from your research documents...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Reasoning over your research context..."):
            response = chain.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": session_id}},
            )

        answer = response.get("answer", "")
        source_pairs = extract_unique_sources(response)
        sources = [{"source": source, "page": page} for source, page in source_pairs]

        log_turn_to_reference_db(
            reference_db_path=reference_db,
            session_id=session_id,
            user_query=prompt,
            assistant_answer=answer,
            sources=source_pairs,
        )

        st.markdown(answer)
        if show_sources and sources:
            with st.expander("Sources"):
                st.markdown(format_sources_md(sources))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


if __name__ == "__main__":
    main()
