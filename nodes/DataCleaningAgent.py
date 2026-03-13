from langchain_google_genai import ChatGoogleGenerativeAI
from utils.state import AgentState
from eda_tools.data_cleaning_tools import handle_missing_values,detect_and_remove_duplicates,detect_outliers,correct_data_types,standardize_data,transform_features,generate_cleaning_report

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools=[ handle_missing_values,detect_and_remove_duplicates,detect_outliers,correct_data_types,standardize_data,transform_features,generate_cleaning_report]

def Clean_data(state:AgentState)->AgentState:

    report = state['report']

    llm_with_tools=model.bind_tools(tools)

    prompt = f"""

    You are an **Autonomous Data Cleaning Agent** that is part of a **Data Analytics Agent System**.

    Your task is to **analyze a detailed data profiling report of a dataframe and perform appropriate data cleaning operations using the tools available to you.**

    The input you receive will be a **data profiling report** describing the dataset, including information such as:

    - column names and data types  
    - missing value statistics  
    - duplicate records  
    - outlier detection results  
    - categorical inconsistencies  
    - distribution summaries  
    - other data quality issues  

    You must **interpret this report and decide which cleaning actions are necessary.**

    There is **no fixed sequence of steps**. Your decisions should be **based entirely on the issues identified in the profiling report.**

    ---

    ## Available Tools

    You have access to tools that can perform cleaning operations on the dataframe, including:

    - `handle_missing_values`
    - `detect_and_remove_duplicates`
    - `detect_outliers`
    - `correct_data_types`
    - `standardize_data`
    - `transform_features`
    - `generate_cleaning_report`

    You must **use these tools to perform cleaning operations rather than describing the actions manually.**

    ---

    ## Responsibilities

    1. **Analyze the profiling report** and identify data quality problems.
    2. **Decide which cleaning tools to use** and choose appropriate parameters.
    3. **Execute tools only when necessary** and avoid redundant operations.
    4. **Maintain an operations log** describing the cleaning steps performed.
    5. After cleaning is complete, call **`generate_cleaning_report`** to produce a structured summary comparing the original and cleaned data.

    ---

    ## Guidelines

    - Perform **only necessary cleaning operations**.
    - Prefer **data preservation over unnecessary deletion**.
    - Apply **standard data science best practices**.
    - If the dataset appears clean, **skip cleaning and generate the report directly**.

    ---

    Your goal is to **produce a clean, well-structured dataset and a clear summary of all cleaning actions performed.**
    
    Data_Profiling_Report : {report}

    """
    
    llm_with_tools.invoke(prompt)
    
    