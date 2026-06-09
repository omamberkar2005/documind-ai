import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── 1. LOAD PDF ───────────────────────────────────────────────
loader = PyPDFLoader(r"C:\Users\DELL\OneDrive\Documents\kdramas.pdf")
documents = loader.load()
for i, doc in enumerate(documents):
    print(f"Page {i+1} preview: {doc.page_content[:100]}")
print(f"Loaded {len(documents)} pages")

# ── 2. CHUNK ──────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# ── 3. EMBED + VECTOR DB ──────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
retriever = vectordb.as_retriever(search_kwargs={"k": 6})
print("Vector DB created ✓")

# ── 4. LLM ────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

# ── 5. PROMPT ─────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
If the answer isn't in the context, say "I don't know based on the document."

Context:
{context}

Question:
{question}
""")

# ── 6. CHAIN ──────────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── 7. ASK QUESTIONS ──────────────────────────────────────────
print("\n📄 RAG System Ready. Type 'quit' to exit.\n")

while True:
    query = input("Your question: ")
    if query.lower() == "quit":
        break
    
    answer = chain.invoke(query)
    print(f"\n🤖 Answer:\n{answer}\n")