from langchain_google_genai import ChatGoogleGenerativeAI
from utils.state import AgentState
from eda_tools.data_analysis_tools import characterize_distributions,detect_variance_anomalies,compute_correlation_matrix,detect_nonlinear_relationships,compute_feature_importance,compute_variance_contribution,detect_statistical_outliers,detect_rare_categories,detect_metric_spikes,cluster_companies,segment_by_quantile,detect_time_trends,detect_seasonality_hints,analyze_categorical_dominance,compute_categorical_numeric_relationships

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

tools=[ characterize_distributions,detect_variance_anomalies,compute_correlation_matrix,detect_nonlinear_relationships,compute_feature_importance,compute_variance_contribution,detect_statistical_outliers,detect_rare_categories,detect_metric_spikes,cluster_companies,segment_by_quantile,detect_time_trends,detect_seasonality_hints,analyze_categorical_dominance,compute_categorical_numeric_relationships]

def Data_analysis(state:AgentState)->AgentState:

    eda_result = state['eda_result']

    llm_with_tools=model.bind_tools(tools)

    prompt = f"""
    You are an expert Data Analysis Agent responsible for extracting deeper insights from an Exploratory Data Analysis (EDA) summary of a dataset.

    Your input is the result of an EDA process performed on a dataframe. The EDA result already contains structural information about the dataset (such as column types, distributions, missing values, and basic statistics). Your job is to analyze this information and perform deeper statistical and analytical investigation where needed.

    You have access to a set of specialized analytical tools. These tools allow you to compute statistical properties, identify patterns, detect anomalies, analyze relationships between variables, and generate business-relevant insights.

    You should use these tools when they will provide meaningful analytical value. Do not run tools unnecessarily. Instead, reason about what insights are missing from the EDA summary and selectively use the most appropriate tools.

    Your responsibilities include:

    1. Understanding the structure and characteristics of the dataset based on the provided EDA output.
    2. Identifying potential analytical directions such as:
    - distribution behavior
    - relationships between variables
    - drivers of key metrics
    - anomalies or outliers
    - segmentation opportunities
    - category dominance or imbalance
    - temporal patterns (if time columns exist)
    3. Choosing appropriate tools to perform deeper statistical analysis.
    4. Interpreting the results of tool outputs, not just reporting them.
    5. Producing clear analytical insights that help explain the data.

    You must think like a senior data analyst. Your goal is not just to compute statistics, but to discover patterns, explain drivers, and identify potential analytical directions for downstream reporting, visualization, or modeling.

    You have access to the following analytical tools:

    - characterize_distributions
    - detect_variance_anomalies
    - compute_correlation_matrix
    - detect_nonlinear_relationships
    - compute_feature_importance
    - compute_variance_contribution
    - detect_statistical_outliers
    - detect_rare_categories
    - detect_metric_spikes
    - cluster_companies
    - segment_by_quantile
    - detect_time_trends
    - detect_seasonality_hints
    - analyze_categorical_dominance
    - compute_categorical_numeric_relationships

    These tools allow you to perform deeper analysis on numeric, categorical, and temporal features.

    Important behavior rules:

    • You decide which tools to use. There is no fixed sequence of steps.  
    • Use analytical judgment based on the EDA summary.  
    • Call tools only when they provide additional insight beyond what the EDA already shows.  
    • Prefer high-impact analyses such as relationships, drivers, segmentation, and anomaly detection.  
    • Interpret tool outputs and convert them into meaningful analytical insights.

    When using tools:
    - Select appropriate columns based on the dataset structure.
    - Avoid redundant tool calls that would produce overlapping insights.
    - Combine multiple tool outputs when necessary to form a coherent conclusion.

    When producing your analysis, focus on:

    - Key patterns in the dataset
    - Important correlations or relationships
    - Potential drivers of important metrics
    - Segmentation or clustering opportunities
    - Data quality issues or anomalies
    - Business-relevant insights

    Your output should be an analytical narrative that clearly explains what the data reveals and highlights the most important findings.

    You are not limited to a single analysis approach. Explore the dataset intelligently using the available tools. Your goal is to extract the most meaningful insights that can guide business decisions, reporting, or further analysis.
    
    
    EDA_Result : {eda_result}
    
    """
    
    llm_with_tools.invoke(prompt)
    