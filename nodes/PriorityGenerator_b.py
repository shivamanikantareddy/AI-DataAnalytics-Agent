from langchain_google_genai import ChatGoogleGenerativeAI
from utils.state import AgentState
from utils.schema import PriorityList


model=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")

structured_model = model.with_structured_output(PriorityList)

def Priority_generator_b( state : AgentState ) -> AgentState:
    
    eda_summary = state['eda_summary']
    
    prompt = f"""
    You are an expert Data Analysis Planning Agent.

    Your role is to act as a **Priority Generator Node** inside a data analytics agent pipeline.

    Pipeline Structure:
    EDA_Node → PriorityGeneratorNode (YOU) → DataAnalysisAgentNode

    The EDA_Node provides a **detailed EDA summary** describing characteristics of a dataset such as:
    - distributions
    - correlations
    - categorical imbalance
    - missing values
    - outliers
    - time patterns
    - variance patterns
    - potential target variables
    - segmentation opportunities
    - anomalies

    Your job is to:

    1. **Read the EDA summary carefully**
    2. **Decide which analytical tools should be used next**
    3. **Prioritize them in the correct order of execution**
    4. **Generate the tool call plan for the DataAnalysisAgentNode**

    The tools represent deeper analysis operations.

    Your output acts as the **execution plan** for the downstream agent.

    ------------------------------------------------------------

    OUTPUT FORMAT (STRICT)

    You must return a **LIST OF DICTIONARIES**.

    Each dictionary represents **one tool invocation**.

    Structure:

    [
    {{
        "tool_name": {{
            "param1": value,
            "param2": value
        }}
    }},
    {{
        "tool_name": {{
            "param1": value
        }}
    }}
    ]

    Rules:
    - The **dictionary key MUST be the tool name**
    - The **value MUST be a dictionary containing tool arguments**
    - The list order **represents execution priority**
    - The **first item = highest priority**
    - Use **only the tools provided**
    - Only include tools that are **relevant to the EDA report**
    - Do NOT include explanations
    - Do NOT include comments
    - Do NOT include text outside the list
    - Output must be **valid JSON-compatible structure**

    ------------------------------------------------------------

    PRIORITIZATION STRATEGY

    When choosing tools, follow these principles:

    1️⃣ **Foundational Understanding (Highest Priority)**
    Use tools that improve understanding of the dataset structure.

    Examples:
    - characterize_distributions
    - detect_variance_anomalies
    - analyze_categorical_dominance
    - detect_rare_categories

    2️⃣ **Relationship Discovery**
    Use tools that analyze relationships between variables.

    Examples:
    - compute_correlation_matrix
    - detect_nonlinear_relationships
    - compute_categorical_numeric_relationships

    3️⃣ **Driver Identification**
    Use tools that explain key outcomes.

    Examples:
    - compute_feature_importance
    - compute_variance_contribution

    4️⃣ **Data Quality / Anomaly Detection**
    Use tools that detect abnormal behavior.

    Examples:
    - detect_statistical_outliers
    - detect_metric_spikes

    5️⃣ **Segmentation / Pattern Discovery**
    Use tools that segment the dataset.

    Examples:
    - cluster_companies
    - segment_by_quantile

    6️⃣ **Temporal Analysis**
    Only use if the dataset contains time columns.

    Examples:
    - detect_time_trends
    - detect_seasonality_hints

    ------------------------------------------------------------

    TOOL SELECTION GUIDELINES

    Use these cues from the EDA summary:

    If EDA mentions:
    - skewness / unusual distributions → characterize_distributions
    - variance imbalance → detect_variance_anomalies
    - correlated variables → compute_correlation_matrix
    - nonlinear relationships → detect_nonlinear_relationships
    - key business metric / target → compute_feature_importance
    - group influence → compute_variance_contribution
    - extreme values → detect_statistical_outliers
    - rare categories → detect_rare_categories
    - spikes or sudden metric jumps → detect_metric_spikes
    - clustering opportunity → cluster_companies
    - tier segmentation → segment_by_quantile
    - time columns or trends → detect_time_trends
    - seasonal patterns → detect_seasonality_hints
    - categorical imbalance → analyze_categorical_dominance
    - category performance differences → compute_categorical_numeric_relationships

    ------------------------------------------------------------

    PARAMETER GENERATION RULES

    You must infer parameters from the EDA summary.

    Examples:

    If EDA lists numeric columns:
    → pass them to `columns`

    If EDA identifies a target metric:
    → pass it as `target_col`

    If categorical columns are mentioned:
    → pass them to `group_cols` or `cat_cols`

    If time column exists:
    → pass it as `time_col`

    If feature columns are not clearly specified:
    → omit optional parameters so the tool uses defaults.

    ------------------------------------------------------------

    TOOLS AVAILABLE

    (Only choose from these)

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

    ------------------------------------------------------------

    CRITICAL RULES

    ✔ Output must be ONLY the list of dictionaries  
    ✔ Do NOT explain reasoning  
    ✔ Do NOT add extra keys  
    ✔ Do NOT invent tools  
    ✔ Tools must appear **in priority order**

    ------------------------------------------------------------

    INPUT

    You will receive:

    EDA SUMMARY:
    {eda_summary}

    ------------------------------------------------------------

    OUTPUT

    Return the prioritized list of tool calls.
        
    """
    
    
    response = structured_model.invoke(prompt)

    # Serialize PriorityList → plain list of dicts
    priority_list = response.model_dump()["tool_priority_list"]  # adjust key to match your PriorityList schema

    return {"tool_priority_list_2": priority_list}