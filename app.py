import os
from pathlib import Path
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------
# UI CONFIG
# -----------------------------
st.set_page_config(page_title="Conchita RAG (Groq)", page_icon="🐚", layout="wide")
st.title("🐚 Conchita RAG Assistant (Groq + FAISS)")
st.caption("PDF ya en el repo. Tú pregunta, yo hago el resto. 😌")


# -----------------------------
# HELPERS
# -----------------------------
def build_llm(api_key: str, model: str, temperature: float):
    os.environ["GROQ_API_KEY"] = api_key
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )


def find_repo_pdfs() -> list[str]:
    """Find PDFs in the same folder (and subfolders if you want)."""
    pdfs = sorted([str(p) for p in Path(".").glob("*.pdf")])
    return pdfs


def load_texts_from_pdf_paths(pdf_paths: list[str]) -> list[str]:
    texts = []
    for path in pdf_paths:
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        full_text = "\n".join(d.page_content for d in docs)
        texts.append(full_text)
    return texts


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


@st.cache_resource(show_spinner=False)
def build_vectorstore_cached(pdf_paths: tuple, embedding_model_name: str, chunk_size: int, chunk_overlap: int):
    texts = load_texts_from_pdf_paths(list(pdf_paths))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vs = FAISS.from_texts(chunks, embedding=embeddings)
    return vs, len(chunks)


def make_rag_chain(llm, retriever):
    system_prompt = (
        "You are a helpful virtual assistant answering general questions about a company's services.\n"
        "Use the retrieved context to answer the question.\n"
        "If you don't know, say you don't know.\n"
        "Keep the answer concise.\n"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: {input}\n\nContext:\n{context}"),
        ]
    )

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "input": RunnablePassthrough(),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return chain


# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
with st.sidebar:
    st.header("⚙️ Configuración")

    api_key = st.text_input("Groq API Key", type="password", help="Pega tu GROQ_API_KEY aquí.")

    model = st.selectbox(
        "Modelo Groq",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ],
        index=0,
    )

    temperature = st.slider("Temperatura", 0.0, 1.5, 0.7, 0.1)

    st.divider()
    st.subheader("📚 Embeddings / Index")
    embedding_model = st.selectbox(
        "Modelo de embeddings",
        options=[
            "BAAI/bge-large-en-v1.5",
            "sentence-transformers/all-MiniLM-L6-v2",
        ],
        index=0,
    )

    chunk_size = st.slider("Chunk size", 200, 1500, 500, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 50, 10)
    top_k = st.slider("Top-k retrieval", 1, 12, 6, 1)

    st.divider()
    st.subheader("📄 PDFs detectados")
    repo_pdfs = find_repo_pdfs()
    if repo_pdfs:
        for p in repo_pdfs:
            st.write(f"✅ {p}")
    else:
        st.warning("No se ha encontrado ningún PDF en el repo (misma carpeta que app.py).")


# -----------------------------
# STATE INIT
# -----------------------------
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# AUTO INDEX BUILD
# -----------------------------
if api_key and repo_pdfs and st.session_state.rag_chain is None:
    with st.spinner("Cargando PDF(s) del repo y construyendo índice…"):
        vs, n_chunks = build_vectorstore_cached(
            pdf_paths=tuple(repo_pdfs),
            embedding_model_name=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

        llm = build_llm(api_key=api_key, model=model, temperature=temperature)
        st.session_state.rag_chain = make_rag_chain(llm, retriever)

    st.success(f"Índice listo ✅ Chunks creados: {n_chunks}")


# -----------------------------
# CHAT UI
# -----------------------------
st.subheader("💬 Chat")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Escribe tu pregunta sobre el PDF del repo…")

if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    if not api_key:
        with st.chat_message("assistant"):
            st.error("Pon tu Groq API Key en el sidebar.")
    elif not repo_pdfs:
        with st.chat_message("assistant"):
            st.error("No hay PDFs en el repo. Sube uno al mismo directorio que app.py.")
    elif st.session_state.rag_chain is None:
        with st.chat_message("assistant"):
            st.error("El índice aún no está listo. Revisa el sidebar o recarga.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Pensando…"):
                try:
                    answer = st.session_state.rag_chain.invoke(user_q)
                except Exception as e:
                    answer = f"Error llamando al modelo o al chain: {e}"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
