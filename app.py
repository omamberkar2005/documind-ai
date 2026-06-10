import os
import shutil
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Main container */
.block-container {
    max-width: 800px !important;
    padding: 2rem 2rem !important;
}

/* Title styling */
h1 {
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    text-align: center;
    margin-bottom: 0 !important;
}

/* Caption */
.stApp p {
    color: #94a3b8 !important;
}

/* File uploader */
.stFileUploader {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(167,139,250,0.3) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
}

.stFileUploader label {
    color: #c4b5fd !important;
    font-weight: 500 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
    transform: translateY(-1px) !important;
}

/* Chat messages - user */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(37,99,235,0.2)) !important;
    border: 1px solid rgba(167,139,250,0.3) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    margin: 0.5rem 0 !important;
}

/* Chat messages - assistant */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    margin: 0.5rem 0 !important;
}

/* Chat input */
/* Chat input container */
.stChatInput > div,
div[data-testid="stChatInput"] > div {
    background: #1e1b4b !important;
    border: 1px solid rgba(167,139,250,0.5) !important;
    border-radius: 16px !important;
}

/* The actual textarea */
.stChatInput textarea,
div[data-testid="stChatInput"] textarea {
    background: #1e1b4b !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #a78bfa !important;
}

/* Placeholder text */
.stChatInput textarea::placeholder {
    color: #6b7280 !important;
    -webkit-text-fill-color: #6b7280 !important;
}

/* Send button */
.stChatInput button {
    background: linear-gradient(90deg, #7c3aed, #2563eb) !important;
    border-radius: 10px !important;
    border: none !important;
}

/* Success/info boxes */
.stSuccess {
    background: rgba(16,185,129,0.1) !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
    border-radius: 12px !important;
    color: #6ee7b7 !important;
}

.stInfo {
    background: rgba(96,165,250,0.1) !important;
    border: 1px solid rgba(96,165,250,0.3) !important;
    border-radius: 12px !important;
    color: #93c5fd !important;
}

.stWarning {
    background: rgba(245,158,11,0.1) !important;
    border: 1px solid rgba(245,158,11,0.3) !important;
    border-radius: 12px !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #7c3aed !important;
}

/* Expander (sources) */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 10px !important;
    color: #a78bfa !important;
    font-size: 0.85rem !important;
}

.streamlit-expanderContent {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 0 0 10px 10px !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
}

/* Text colors */
.stMarkdown p, .stText { color: #e2e8f0 !important; }
label { color: #c4b5fd !important; }

/* Divider */
hr {
    border-color: rgba(167,139,250,0.2) !important;
}

/* Caption text */
.stCaption { color: #64748b !important; }

/* Fix all text inside chat messages */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] ol,
[data-testid="stChatMessage"] ul,
[data-testid="stChatMessage"] span {
    color: #e2e8f0 !important;
}

/* Fix markdown lists specifically */
[data-testid="stChatMessage"] .stMarkdown {
    color: #e2e8f0 !important;
}

</style>
""", unsafe_allow_html=True)

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(page_title="AskMyPDF", page_icon="🧠", layout="centered")

# ── HEADER ────────────────────────────────────────────────────
st.markdown("<h1>🧠 AskMyPDF</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94a3b8; margin-bottom:2rem;'>Upload any PDF and have a conversation with it</p>", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "sources" not in st.session_state:
    st.session_state.sources = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "first_page" not in st.session_state:
    st.session_state.first_page = ""
if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""

# ── CLEAR BUTTON ──────────────────────────────────────────────
if st.session_state.chain:
    col1, col2, col3 = st.columns([2,2,1])
    with col3:
        if st.button("🔄 New PDF"):
            st.session_state.chain = None
            st.session_state.chat_history = []
            st.session_state.sources = []
            st.session_state.retriever = None
            st.session_state.first_page = ""
            st.session_state.doc_name = ""
            if os.path.exists("./chroma_db"):
                try:
                    shutil.rmtree("./chroma_db")
                except PermissionError:
                    st.warning("Stop the app, delete the chroma_db folder manually, then restart.")
                    st.stop()
            st.rerun()

# ── PDF UPLOAD ────────────────────────────────────────────────
if not st.session_state.chain:
    uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        st.markdown(f"<p style='color:#a78bfa; font-size:0.9rem;'>📄 {uploaded_file.name}</p>", unsafe_allow_html=True)

        with st.spinner("✨ Reading and indexing your document..."):
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            loader = PyMuPDFLoader(temp_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
            chunks = splitter.split_documents(documents)

            st.session_state.first_page = documents[0].page_content
            st.session_state.doc_name = uploaded_file.name

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db")
            st.session_state.retriever = vectordb.as_retriever(search_kwargs={"k": 10})

            llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), temperature=0.2)

            prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions about a document.
Use ONLY the context below to answer. Be concise and clear.
If the answer is not in the context, say: "I couldn't find that in the document."
Never invent information.

Document First Page:
{first_page}

Context:
{context}

Question:
{question}

Answer:
""")

            def format_docs(docs):
                st.session_state.sources = docs
                return "\n\n".join(doc.page_content for doc in docs)

            first_page_content = st.session_state.first_page
            retriever = st.session_state.retriever

            st.session_state.chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough(),
                    "first_page": lambda _: first_page_content
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            os.remove(temp_path)

        st.success(f"✅ Ready! Indexed {len(chunks)} chunks from {len(documents)} pages.")
        st.rerun()

# ── CHAT INTERFACE ────────────────────────────────────────────
if st.session_state.chain:

    # Doc info bar
    st.markdown(f"""
    <div style='background:rgba(124,58,237,0.15); border:1px solid rgba(124,58,237,0.3);
    border-radius:12px; padding:0.6rem 1rem; margin-bottom:1rem; font-size:0.85rem; color:#c4b5fd;'>
    📄 <strong>{st.session_state.doc_name}</strong> — Ask me anything about this document
    </div>
    """, unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📎 View Sources"):
                    for doc in msg["sources"]:
                        page = doc.metadata.get("page", "?")
                        st.caption(f"**Page {int(page)+1}:** {doc.page_content[:200]}...")

    # Input
    query = st.chat_input("Ask anything about your document...")

    if query:
        with st.chat_message("user"):
            st.write(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(query)
                sources = st.session_state.sources.copy()
            st.write(answer)
            if sources:
                with st.expander("📎 View Sources"):
                    for doc in sources:
                        page = doc.metadata.get("page", "?")
                        st.caption(f"**Page {int(page)+1}:** {doc.page_content[:200]}...")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

else:
    st.markdown("""
    <div style='text-align:center; padding:3rem 1rem; color:#475569;'>
        <div style='font-size:3rem; margin-bottom:1rem;'>📄</div>
        <p style='font-size:1.1rem; color:#64748b;'>Upload a PDF above to get started</p>
        <p style='font-size:0.85rem; color:#475569;'>Supports research papers, books, reports, and more</p>
    </div>
    """, unsafe_allow_html=True)