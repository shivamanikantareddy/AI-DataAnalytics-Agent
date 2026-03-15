from langchain_google_genai import ChatGoogleGenerativeAI
from utils.state import AgentState


model=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")


def Generate_summary( state : AgentState ) -> AgentState:

    cleaning_summary = state['cleaning_summary']
    eda_results = state['eda_results']
    analysis_results = state['analysis_results']

    prompt = f"""
    You are a Summary Generator Agent in a Data Analytics system.

    Your responsibility is to generate a comprehensive summary report based on the results of data cleaning and analysis.

    INPUT:
    You will receive the following information:
    1. A summary of the data cleaning performed on the data frae and their outcomes.
    2. A summary of the exploratory data analysis (EDA) results, including key insights about distributions, correlations, trends, etc.
    3. Detailed results from various analyses performed on the data.

    YOUR TASK:
    1. Synthesize all the provided information into a coherent and comprehensive summary report.
    2. Highlight key findings from both the cleaning and analysis phases.
    3. keep the language simple and accessible, as the report will be used by a RAG-based chatbot to answer queries from users who may not have technical expertise.
    4. Ensure that the summary is well-structured, with clear sections for cleaning results, EDA insights, and analysis findings.
    5. The summary should be detailed enough to provide a clear understanding of the data's condition and the insights derived from it, while also being concise and easy to understand for non-technical users.

    OUTPUT:
    Generate a well-structured and highly detailed summary report that will be used as the data for a rag based chatbot to answer all the queries of the users who may not have technical expertise.
    
    Note: The summary should be just text and should not include any code fromats or markdown syntax, just use sentences and paragraphs. As it will be directly fed into a rag-based chatbot for answering user queries.
    
    ## Cleaning Summary : 
    {cleaning_summary}
    
    ## EDA Results :
    {eda_results}
    
    ## Analysis Results :
    {analysis_results}
    
     """

    response = model.invoke(prompt)
    
    filepath = "./uploads/Analytics_summary.txt"

    with open(filepath, "w") as file:
        file.write(response)



