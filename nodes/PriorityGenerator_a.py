from langchain_google_genai import ChatGoogleGenerativeAI
from utils.state import AgentState
from utils.schema import PriorityList


model=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")

structured_model = model.with_structured_output(PriorityList)

def Priority_generator_a( state : AgentState ) -> AgentState:
    
    report = state['report']
    
    prompt = f"""
    # ROLE
    You are an expert Data Cleaning Planner inside a Data Analytics Agent system.

    Your job is to analyze a **Data Profiling Report of a DataFrame** and generate a
    **prioritized list of tools** that should be executed by a downstream **DataCleaningAgent**.

    You are part of the following pipeline:

    DataProfilingNode → PriorityGeneratorNode → DataCleaningAgent

    Your responsibility is to:
    1. Analyze the profiling report.
    2. Identify which data cleaning tools are necessary.
    3. Determine the **correct execution order (priority)**.
    4. Generate a **list of tool calls with parameters** for the DataCleaningAgent.

    ---

    # INPUT

    You will receive a **detailed data profiling report** describing a DataFrame.
    The report may contain information such as:

    - Column names
    - Data types
    - Missing value statistics
    - Duplicate row statistics
    - Outlier statistics
    - Text inconsistencies
    - Categorical inconsistencies
    - Numeric distributions
    - Potential type mismatches
    - Feature transformation suggestions

    Use this information to determine the correct **data cleaning workflow**.

    ---

    # AVAILABLE TOOLS

    You may only select from the following tools.

    ---

    ## `handle_missing_values`
    Detect and handle missing values in a DataFrame.

    | Parameter        | Type         | Notes                                      |
    |------------------|--------------|--------------------------------------------|
    | `strategy`       | `str`        | `'mean'`, `'median'`, `'mode'`, `'ffill'`, `'bfill'`, `'drop_rows'`, `'drop_cols'`, `'constant'` |
    | `columns`        | `list[str]`  | Optional                                   |
    | `fill_value`     | any          | Used when `strategy='constant'`            |
    | `drop_threshold` | `float`      | Used for `drop_cols`                       |

    ---

    ## `detect_and_remove_duplicates`
    Detect and optionally remove duplicate rows.

    | Parameter  | Type        | Notes                              |
    |------------|-------------|------------------------------------|
    | `subset`   | `list[str]` | Optional                           |
    | `keep`     | `str`       | `'first'`, `'last'`, or `False`    |
    | `remove`   | `bool`      |                                    |

    ---

    ## `detect_outliers`
    Detect outliers using IQR, Z-score, or Isolation Forest.

    | Parameter         | Type        | Notes                                        |
    |-------------------|-------------|----------------------------------------------|
    | `method`          | `str`       | `'iqr'`, `'zscore'`, `'isolation_forest'`    |
    | `columns`         | `list[str]` |                                              |
    | `action`          | `str`       | `'flag'`, `'remove'`, `'cap'`                |
    | `z_threshold`     | `float`     |                                              |
    | `iqr_multiplier`  | `float`     |                                              |
    | `contamination`   | `float`     |                                              |

    ---

    ## `fix_dtypes`
    Convert and correct column data types.

    | Parameter            | Type        | Notes                        |
    |----------------------|-------------|------------------------------|
    | `numeric_cols`       | `list[str]` | Optional                     |
    | `date_cols`          | `list[str]` | Optional                     |
    | `categorical_cols`   | `list[str]` | Optional                     |
    | `auto_detect`        | `bool`      | Default `True`               |
    | `date_format`        | `str`       | Optional                     |
    | `category_threshold` | `float`     | Default `0.10`               |
    | `downcast_numerics`  | `bool`      | Default `True`               |
    | `inplace`            | `bool`      | Default `False`              |

    ---

    ## `standardize_data`
    Standardize text, categorical, and fuzzy-matched values.

    | Parameter          | Type        | Notes    |
    |--------------------|-------------|----------|
    | `text_cols`        | `list[str]` | Optional |
    | `categorical_cols` | `list[str]` | Optional |
    | `fuzzy_map`        | `dict`      | Optional |
    | `fuzzy_threshold`  | `int`       | Optional |

    ---

    ## `transform_features`
    Apply feature transformations.

    | Parameter            | Type        | Notes                                                                 |
    |----------------------|-------------|-----------------------------------------------------------------------|
    | `method`             | `str`       | `'standard_scaler'`, `'minmax_scaler'`, `'log'`, `'onehot'`, `'label_encoder'`, `'featuretools'`, `'featuretools_full'` |
    | `columns`            | `list[str]` |                                                                       |
    | `log_shift`          | `float`     | Optional                                                              |
    | `ohe_drop`           | `str`       | Optional                                                              |
    | `entity_id_col`      | `str`       | Optional                                                              |
    | `ft_agg_primitives`  | `list[str]` | Optional                                                              |
    | `ft_trans_primitives`| `list[str]` | Optional                                                              |
    | `ft_max_depth`       | `int`       | Optional                                                              |

    ---

    # PRIORITY RULES

    Determine the correct **execution order** based on the profiling report findings.
    Only include tools that are **actually required** based on the profiling report.
    priority should be based on standard data cleaning best practices and the specific issues identified in the profiling report.
    The priority order should be in decreasing order of importance, meaning the most critical cleaning steps should be listed first.
    ---

    # DECISION RULES

    Use the profiling report to decide which tools to include:

    | Condition                                              | Include Tool                      |
    |--------------------------------------------------------|-----------------------------------|
    | Columns have incorrect or inconsistent types           | `fix_dtypes`              |
    | Missing values exist                                   | `handle_missing_values`           |
    | Duplicate rows are detected                            | `detect_and_remove_duplicates`    |
    | Numeric outliers are detected                          | `detect_outliers`                 |
    | Inconsistent casing, spacing, or similar labels exist  | `standardize_data`                |
    | Categorical encoding or scaling is needed              | `transform_features`              |

    ---

    # OUTPUT FORMAT

    Return a **Python-style list of dictionaries**.

    Each dictionary must follow this structure:
    ```python
    {{
    "tool_name": {{
        "parameter1": value,
        "parameter2": value
    }}
    }}
    ```
    
    ###Example Output
    ```python
    [
    {{
        "fix_dtypes": {{
            "auto_detect": true
        }}
    }},
    {{
        "handle_missing_values": {{
            "strategy": "median",
            "columns": ["age", "salary"]
        }}
    }},
    {{
        "detect_and_remove_duplicates": {{
            "subset": ["customer_id"],
            "keep": "first",
            "remove": true
        }}
    }},
    {{
        "detect_outliers": {{
            "method": "iqr",
            "columns": ["salary"],
            "action": "cap"
        }}
    }},
    {{
        "standardize_data": {{
            "text_cols": ["city", "country"],
            "categorical_cols": ["gender"]
        }}
    }}
    ]
    ```

    ---

    # IMPORTANT CONSTRAINTS

    > Strictly follow all rules below. Violations will break the downstream pipeline.

    - Output **ONLY the list of dictionaries** — no explanations, no commentary.
    - Do **NOT** include markdown formatting, prose, or headers in your output.
    - Tools must appear in **correct execution order** as defined in Priority Rules.
    - Include **only tools relevant** to the profiling report findings.
    - Ensure all parameters **exactly match** the tool definitions above.
    - **Omit optional parameters** if they are not required by the report.

    ---

    # INPUT DATA PROFILING REPORT
    ```
    {report}
    ```

    Analyze the report above and produce the prioritized cleaning plan list.
        
    """
    
    
    response = structured_model.invoke(prompt)

    # Serialize PriorityList → plain list of dicts
    priority_list = response.model_dump()["tool_priority_list"]  # adjust key to match your PriorityList schema

    return {"tool_priority_list_1": priority_list}