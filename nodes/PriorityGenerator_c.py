from langchain_google_genai import ChatGoogleGenerativeAI
from utils.state import AgentState
from utils.schema import PriorityList


model=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")

structured_model = model.with_structured_output(PriorityList)

def Priority_generator_c( state : AgentState ) -> AgentState:
    
    eda_summary = state['eda_summary']
    analysis_result = state["analysis_results"]
    
    prompt = f"""
    ````
    # SYSTEM PROMPT — Priority Generator Node

    ## Role & Objective

    You are the **Priority Generator** in a multi-agent data analytics pipeline. Your sole responsibility is to act as the intelligent bridge between the **Data Analysis Agent** and the **Data Visualization Agent**.

    You receive:
    1. **EDA Summary** — A detailed exploratory data analysis report of the dataset, including column names, data types, distributions, missing values, outlier statistics, cardinality, skewness, correlations, and any temporal or geospatial metadata detected.
    2. **Analysis Result** — The structured output of the Data Analysis Agent, which may include findings such as significant correlations, regression results, group comparisons, clustering outcomes, time-series decompositions, anomaly detections, or statistical significance tests.

    Your task is to **intelligently select the most relevant visualization tools** from the available toolkit and **return them as a prioritized list of dictionaries**, where each dictionary maps a single tool name to its input parameters. The list must be ordered from **highest priority to lowest priority** — the visualization agent will execute tools in this order.

    ---

    ## Input Format

    You will receive two blocks:
    ````
    ### EDA SUMMARY
    <detailed EDA summary text>

    ### ANALYSIS RESULT
    <structured output from the Data Analysis Agent>
    ````

    ---

    ## Available Tools & Their Parameters

    Below is the complete reference of tools available to the Data Visualization Agent. You must only select tools from this list.

    ### Univariate — Numeric
    - `create_histogram` → `{{column, bins, kde, title, color, save_path}}`
    - `create_kde_plot` → `{{column, group_by, title, save_path}}`
    - `create_box_plot` → `{{column, title, color, save_path}}`
    - `create_violin_plot` → `{{column, title, color, save_path}}`

    ### Univariate — Categorical
    - `create_frequency_bar_chart` → `{{column, top_n, title, color, save_path}}`
    - `create_pie_chart` → `{{column, top_n, title, save_path}}`

    ### Bivariate — Numeric vs Numeric
    - `create_scatter_plot` → `{{x_column, y_column, color_column, size_column, title, save_path}}`
    - `create_regression_plot` → `{{x_column, y_column, title, save_path}}`
    - `create_regression_analysis_visualization` → `{{x_columns, y_column, title, save_path}}`
    - `create_residual_plot` → `{{x_column, y_column, title, save_path}}`
    - `create_bubble_chart` → `{{x_column, y_column, size_column, color_column, title, save_path}}`
    - `create_large_dataset_scatter_aggregation` → `{{x_column, y_column, title, save_path}}`
    - `create_large_dataset_density_visualization` → `{{x_column, y_column, title, save_path}}`
    - `create_scalable_heatmap` → `{{x_column, y_column, bins, title, save_path}}`

    ### Bivariate — Categorical vs Numeric
    - `create_grouped_bar_chart` → `{{x_column, y_column, group_column, title, save_path}}`
    - `create_box_plot_by_category` → `{{x_column, y_column, title, save_path}}`
    - `create_violin_plot_by_category` → `{{x_column, y_column, title, save_path}}`
    - `create_distribution_comparison_chart` → `{{column, group_column, title, save_path}}`
    - `create_statistical_significance_visualization` → `{{column, group_column, title, save_path}}`

    ### Bivariate — Categorical vs Categorical
    - `create_categorical_comparison_chart` → `{{col1, col2, title, save_path}}`
    - `create_stacked_bar_chart` → `{{x_column, y_column, stack_column, normalize, title, save_path}}`

    ### Multivariate
    - `create_pair_plot` → `{{columns, hue, title, save_path}}`
    - `create_correlation_heatmap` → `{{columns, method, title, save_path}}`
    - `create_correlation_matrix_heatmap` → `{{columns, method, title, save_path}}`
    - `create_parallel_coordinates_plot` → `{{columns, color_column, title, save_path}}`
    - `create_radar_chart` → `{{category_column, value_columns, title, save_path}}`
    - `create_grouped_scatter_plot` → `{{x_column, y_column, group_column, title, save_path}}`
    - `create_cluster_heatmap` → `{{columns, title, save_path}}`

    ### Time Series
    - `create_time_series_line_chart` → `{{date_column, value_column, group_column, title, save_path}}`
    - `create_moving_average_chart` → `{{date_column, value_column, windows, title, save_path}}`
    - `create_seasonal_decomposition_plot` → `{{date_column, value_column, period, model, title, save_path}}`
    - `create_time_series_comparison_chart` → `{{date_column, value_columns, title, save_path}}`
    - `create_time_series_area_chart` → `{{date_column, value_column, title, save_path}}`
    - `create_interactive_time_series` → `{{date_column, value_columns, title, save_path}}`

    ### Dimensionality Reduction
    - `create_pca_visualization` → `{{columns, n_components, color_column, title, save_path}}`
    - `create_tsne_plot` → `{{columns, perplexity, color_column, title, save_path}}`
    - `create_umap_plot` → `{{columns, n_neighbors, min_dist, color_column, title, save_path}}`

    ### Hierarchical / Proportional
    - `create_treemap` → `{{path_columns, value_column, title, save_path}}`
    - `create_sunburst_chart` → `{{path_columns, value_column, title, save_path}}`
    - `create_dendrogram` → `{{columns, method, orientation, title, save_path}}`

    ### Geospatial
    - `create_choropleth_map` → `{{location_column, value_column, location_mode, title, save_path}}`
    - `create_geospatial_scatter_map` → `{{lat_column, lon_column, value_column, hover_columns, title, save_path}}`
    - `create_location_density_heatmap` → `{{lat_column, lon_column, value_column, title, save_path}}`
    - `create_regional_comparison_map` → `{{location_column, value_column, scope, title, save_path}}`

    ### Network / Graph
    - `create_node_link_graph` → `{{source_column, target_column, weight_column, layout, title, save_path}}`
    - `create_dependency_graph` → `{{source_column, target_column, title, save_path}}`
    - `create_relationship_network_graph` → `{{source_column, target_column, weight_column, title, save_path}}`

    ### Interactive
    - `create_interactive_scatter_plot` → `{{x_column, y_column, color_column, size_column, hover_columns, title, save_path}}`
    - `create_hover_enabled_bar_chart` → `{{x_column, y_column, color_column, title, save_path}}`
    - `create_zoomable_heatmap` → `{{x_column, y_column, value_column, title, save_path}}`

    ### Data Quality
    - `create_missing_value_heatmap` → `{{title, save_path}}`
    - `create_outlier_detection_plot` → `{{columns, method, threshold, title, save_path}}`
    - `create_distribution_comparison_before_after` → `{{columns, title, save_path}}`

    ---

    ## Prioritization Rules

    Apply these rules in the given order when deciding which tools to select and how to rank them:

    1. **Data Quality First** — If the EDA Summary reports significant missing values or outliers, always prioritize `create_missing_value_heatmap` and `create_outlier_detection_plot` at the top of the list. These give the analyst a clean picture of data health before any deeper analysis.

    2. **Honor Analysis Findings** — Any relationship, pattern, or statistical result explicitly surfaced by the Analysis Agent should be visualized. For example:
    - A strong correlation finding → `create_correlation_heatmap` or `create_regression_plot`
    - A group comparison or ANOVA finding → `create_box_plot_by_category` or `create_statistical_significance_visualization`
    - A clustering result → `create_pca_visualization` or `create_cluster_heatmap`
    - A regression model result → `create_regression_analysis_visualization` and `create_residual_plot`

    3. **Distribution Overview for All Key Columns** — For each important numeric column identified in the EDA, include at least one distribution chart (`create_histogram` or `create_kde_plot`). For each important categorical column, include `create_frequency_bar_chart` or `create_pie_chart` (use pie chart only when cardinality ≤ 7).

    4. **Relationship Depth** — After distributions, add bivariate and multivariate charts for relationships flagged in the analysis. Use `create_pair_plot` if 3 or more correlated numeric columns exist.

    5. **Time-Series Precedence** — If a datetime column is present, time-series tools must be included immediately after data quality tools and before other univariate charts.

    6. **Geospatial Precedence** — If latitude/longitude or country/region columns are detected, include relevant geospatial tools before multivariate charts.

    7. **Large Dataset Awareness** — If the EDA Summary indicates row count > 100,000, prefer scalable alternatives: `create_large_dataset_scatter_aggregation`, `create_large_dataset_density_visualization`, `create_scalable_heatmap` over standard scatter and heatmaps.

    8. **No Redundancy** — Do not select two tools that produce substantially the same visualization for the same column(s). For example, do not select both `create_box_plot` and `create_violin_plot` for the same column unless the Analysis Agent specifically points to distributional shape details.

    9. **Omit Irrelevant Tool Categories** — Do not include network/graph tools unless edge-list relationships are present in the data. Do not include dimensionality reduction tools unless high-dimensional feature analysis was performed by the Analysis Agent.

    10. **Parameter Completeness** — Every parameter you include must be grounded in actual column names and values from the EDA Summary and Analysis Result. Use `null` explicitly for optional parameters that should be left unset. Never fabricate column names.

    ---

    ## Output Format

    Return **only** a valid Python list of dictionaries. Each dictionary must have:
    - **One key**: the exact tool name (string)
    - **One value**: a dictionary of input parameters (key-value pairs using actual column names and settings derived from the inputs)

    The `df` parameter is always managed externally by the Visualization Agent and must **NOT** be included in any parameter dictionary.

    **Format:**
    ```python
    [
        {{
            "tool_name_1": {{
                "param_a": "value_a",
                "param_b": "value_b",
                ...
            }}
        }},
        {{
            "tool_name_2": {{
                "param_x": "value_x",
                ...
            }}
        }},
        ...
    ]
    ```

    ### Rules for the output:
    - Output **only** the Python list. No explanation, no preamble, no markdown — just the raw list.
    - All string values must use double quotes.
    - `null` (not `None`) for optional parameters intentionally left empty.
    - `save_path` should always be set to `null` unless a specific path was provided in the inputs.
    - The `title` parameter should be a concise, human-readable description of what the chart is showing.
    - The list must be ordered from **highest priority (index 0)** to **lowest priority (last index)**.

    ---

    ## Examples

    ### Example 1 — Typical Tabular Dataset

    **EDA Summary (excerpt):**
    - Columns: `age` (numeric), `income` (numeric), `gender` (categorical, 2 unique), `region` (categorical, 5 unique), `purchase_amount` (numeric), `signup_date` (datetime)
    - Missing values: 8% in `income`
    - Outliers detected in `purchase_amount` (IQR method)
    - Strong correlation: `age` ↔ `income` (r = 0.71)

    **Analysis Result (excerpt):**
    - Group comparison: `purchase_amount` differs significantly across `gender` (p < 0.01)
    - OLS regression: `income`, `age` → `purchase_amount` (R² = 0.63)
    - Time trend: `purchase_amount` shows upward trend over `signup_date`

    **Expected Output format:**
    ```python
    [
        {{
            "create_missing_value_heatmap": {{
                "title": "Missing Value Heatmap",
                "save_path": null
            }}
        }},
        {{
            "create_outlier_detection_plot": {{
                "columns": ["purchase_amount"],
                "method": "iqr",
                "threshold": 1.5,
                "title": "Outlier Detection in Purchase Amount",
                "save_path": null
            }}
        }},
        {{
            "create_time_series_line_chart": {{
                "date_column": "signup_date",
                "value_column": "purchase_amount",
                "group_column": null,
                "title": "Purchase Amount Trend Over Time",
                "save_path": null
            }}
        }},
        {{
            "create_histogram": {{
                "column": "age",
                "bins": 30,
                "kde": true,
                "title": "Distribution of Age",
                "color": null,
                "save_path": null
            }}
        }},
        {{
            "create_histogram": {{
                "column": "income",
                "bins": 30,
                "kde": true,
                "title": "Distribution of Income",
                "color": null,
                "save_path": null
            }}
        }},
        {{
            "create_frequency_bar_chart": {{
                "column": "region",
                "top_n": 5,
                "title": "Top Regions by Frequency",
                "color": null,
                "save_path": null
            }}
        }},
        {{
            "create_pie_chart": {{
                "column": "gender",
                "top_n": 2,
                "title": "Gender Distribution",
                "save_path": null
            }}
        }},
        {{
            "create_regression_plot": {{
                "x_column": "age",
                "y_column": "income",
                "title": "Age vs Income with Regression Line",
                "save_path": null
            }}
        }},
        {{
            "create_box_plot_by_category": {{
                "x_column": "gender",
                "y_column": "purchase_amount",
                "title": "Purchase Amount Distribution by Gender",
                "save_path": null
            }}
        }},
        {{
            "create_statistical_significance_visualization": {{
                "column": "purchase_amount",
                "group_column": "gender",
                "title": "Statistical Significance of Purchase Amount Across Gender",
                "save_path": null
            }}
        }},
        {{
            "create_regression_analysis_visualization": {{
                "x_columns": ["income", "age"],
                "y_column": "purchase_amount",
                "title": "OLS Regression Coefficients: Predictors of Purchase Amount",
                "save_path": null
            }}
        }},
        {{
            "create_residual_plot": {{
                "x_column": "income",
                "y_column": "purchase_amount",
                "title": "Residual Plot: Income vs Purchase Amount",
                "save_path": null
            }}
        }},
        {{
            "create_correlation_heatmap": {{
                "columns": ["age", "income", "purchase_amount"],
                "method": "pearson",
                "title": "Correlation Matrix of Numeric Features",
                "save_path": null
            }}
        }}
    ]
    ```

    ---

    Now process the inputs provided below and generate the prioritized tool list.

    ### EDA SUMMARY
    {eda_summary}

    ### ANALYSIS RESULT
    {analysis_result}
    ````
        
    """
    
    
    response = structured_model.invoke(prompt)

    # Serialize PriorityList → plain list of dicts
    priority_list = response.model_dump()["tool_priority_list"]  # adjust key to match your PriorityList schema

    return {"tool_priority_list_3": priority_list}