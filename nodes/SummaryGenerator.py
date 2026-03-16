from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from utils.state import AgentState
from pathlib import Path

VECTORSTORE_PATH = Path("./utils/faiss_index")
ANALYTICS_FILE   = Path("./uploads/Analytics_summary.txt")

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100


model=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


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


def Generate_summary( state : AgentState ) -> AgentState:

    cleaning_summary = state['cleaning_summary']
    eda_results = state['eda_result']
    analysis_results = state['analysis_results']

    prompt = f"""
    You are a Report Generator Agent in a Data Analytics system.

    Your responsibility is to generate a comprehensive report based on the results of data cleaning and analysis.

    INPUT:
    You will receive the following information:
    1. A summary of the data cleaning performed on the data frae and their outcomes.
    2. The result of the exploratory data analysis (EDA) , including key insights about distributions, correlations, trends, etc.
    3. Detailed results from various analyses performed on the data.

    YOUR TASK:
    1. Synthesize all the provided information into a coherent and comprehensive report.
    2. Highlight key findings from both the cleaning and analysis phases.
    3. keep the language simple and accessible, as the report will be used by a RAG-based chatbot to answer queries from users who may not have technical expertise.
    4. Ensure that the report is well-structured, with clear sections for cleaning results, EDA insights, and analysis findings.
    5. The report should be detailed enough to provide a clear understanding of the data's condition and the insights derived from it, while also being concise and easy to understand for non-technical users.

    OUTPUT:
    Generate a well-structured and highly detailed report that will be used as the data for a rag based chatbot to answer all the queries of the users who may not have technical expertise.
    
    Note: The report should be just text and should not include any code fromats or markdown syntax, just use sentences and paragraphs. As it will be directly fed into a rag-based chatbot for answering user queries.
    
    ## Cleaning Summary : 
    {cleaning_summary}
    
    ## EDA Results :
    {eda_results}
    
    ## Analysis Results :
    {analysis_results}
     """

    response = model.invoke(prompt)
    
    # ── Extract text safely regardless of str or list content ────────────────
    content = response.content
    if isinstance(content, list):
        # Gemini sometimes returns [{type: text, text: "..."}] blocks
        summary_text = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    else:
        summary_text = str(content).strip()

    with open(ANALYTICS_FILE, "w", encoding="utf-8") as file:
        file.write(summary_text)
        
    try:
        build_vectorstore(ANALYTICS_FILE)
    except Exception as e:
        # Non-fatal — chat will attempt to build it again on first query
        print(f"[SummaryGenerator] Vectorstore pre-build failed: {e}")



