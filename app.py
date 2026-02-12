import os
import tempfile
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------
# UI CONFIG
# -----------------------------
st.set_page_config(page_title="Conchita RAG (Groq)", page_icon="🐚", layout="wide")
st.title("🐚 Conchita RAG Assistant (Groq + FAISS)")
st.caption("Sube PDFs, indexa, y pregunta. Porque leer manuales es para humanos con tiempo.")


# -----------------------------
# HELPERS
# -----------------------------
def build_llm(api_key: str, model: str, temperature: float):
    # Groq API key via env is what langchain_groq expects
    os.environ["GROQ_API_KEY"] = api_key
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )


def load_pdf_texts(uploaded_files):
    """Return list[str] where each item is full text of a PDF."""
    texts = []
    for uf in uploaded_files:
        # Streamlit uploader gives a file-like buffer; PyMuPDFLoader wants a path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uf.read())
            tmp_path = tmp.name

        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()
        full_text = "\n".join([d.page_content for d in docs])
        texts.append(full_text)

        # Cleanup tmp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return texts


def build_vectorstore(texts, embedding_model_name: str, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vs = FAISS.from_texts(chunks, embedding=embeddings)
    return vs, len(chunks)


def make_rag_chain(llm, retriever):
    system_prompt = (
        "You are a helpful virtual assistant answering general questions about a company's services.\n"
        "Use the following retrieved context to answer the question.\n"
        "If you don't know the answer, say you don't know.\n"
        "Keep the answer concise.\n"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: {input}\n\nContext:\n{context}"),
        ]
    )

    chain = (
        {"context": retriever, "input": RunnablePassthrough()}
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
    temperature = st.slider("Temperatura", min_value=0.0, max_value=1.5, value=0.7, step=0.1)

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
    uploaded_pdfs = st.file_uploader(
        "Sube uno o varios PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    build_index_btn = st.button("🔨 Construir / Rehacer índice", use_container_width=True)


# -----------------------------
# STATE INIT
# -----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# INDEX BUILD
# -----------------------------
if build_index_btn:
    if not api_key:
        st.error("Necesito tu Groq API Key para poder levantar el LLM.")
    elif not uploaded_pdfs:
        st.error("Sube al menos un PDF para indexar.")
    else:
        with st.spinner("Leyendo PDFs y creando índice…"):
            texts = load_pdf_texts(uploaded_pdfs)
            vs, n_chunks = build_vectorstore(
                texts,
                embedding_model_name=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            st.session_state.vectorstore = vs
            retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

            llm = build_llm(api_key=api_key, model=model, temperature=temperature)
            st.session_state.rag_chain = make_rag_chain(llm, retriever)

        st.success(f"Índice listo ✅ Chunks creados: {n_chunks}")


# -----------------------------
# CHAT UI
# -----------------------------
st.subheader("💬 Chat")

# render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# input
user_q = st.chat_input("Escribe tu pregunta sobre los PDFs…")

if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    if st.session_state.rag_chain is None:
        with st.chat_message("assistant"):
            st.error("Primero construye el índice (sube PDFs y pulsa 'Construir / Rehacer índice').")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Pensando (porque los humanos insisten en preguntar cosas) …"):
                try:
                    answer = st.session_state.rag_chain.invoke(user_q)
                except Exception as e:
                    answer = f"Error llamando al modelo o al chain: {e}"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
