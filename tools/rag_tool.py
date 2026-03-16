from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from pathlib import Path


VECTORSTORE_PATH = Path("./utils/faiss_index")

def load_vectorstore():

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )

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
        vectorstore = load_vectorstore()  # ✅ always rebuild with latest data
    except Exception as e:
        return f"Failed to build knowledge base: {e}"

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return "No relevant information found in the knowledge base."

    return "\n\n".join(doc.page_content.strip() for doc in retrieved_docs)