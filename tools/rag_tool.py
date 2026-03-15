from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from pathlib import Path

VECTORSTORE_PATH = Path("./utils/faiss_index")
ANALYTICS_FILE   = Path("./uploads/Analytics_summary.txt")

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
TOP_K         = 4


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


def build_vectorstore(filepath: Path) -> FAISS:
    try:
        loader = TextLoader(str(filepath), encoding="utf-8")
        docs = loader.load()
    except Exception as e:
        raise RuntimeError(f"Failed to load document '{filepath}': {e}") from e

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, _get_embeddings())
    vectorstore.save_local(str(VECTORSTORE_PATH))

    return vectorstore


@tool
def rag_chat_tool(query: str) -> str:
    """
    Retrieves relevant information from the data analysis knowledge base to answer user questions.

    Use this tool for EVERY user question about analysis results, statistics,
    findings, trends, or any content from the analytics report. Always call
    this before formulating your answer.

    Args:
        query: The user's question or search query as a plain string.

    Returns:
        Relevant plain text excerpts from the analytics knowledge base.
        Returns a not-found message if no relevant content exists.
    """
    try:
        vectorstore = build_vectorstore(ANALYTICS_FILE)  # ✅ always rebuild with latest data
    except Exception as e:
        return f"Failed to build knowledge base: {e}"

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return "No relevant information found in the knowledge base."

    return "\n\n".join(doc.page_content.strip() for doc in retrieved_docs)