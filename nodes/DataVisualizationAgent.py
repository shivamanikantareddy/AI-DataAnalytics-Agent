from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from utils.state import AgentState
from tools.data_visualization_tools import create_histogram,create_kde_plot,create_box_plot,create_violin_plot,create_frequency_bar_chart,create_pie_chart,create_scatter_plot,create_regression_plot,create_grouped_bar_chart,create_box_plot_by_category,create_violin_plot_by_category,create_categorical_comparison_chart,create_pair_plot,create_correlation_heatmap,create_bubble_chart,create_grouped_scatter_plot,create_stacked_bar_chart,create_cluster_heatmap,create_parallel_coordinates_plot,create_radar_chart,create_time_series_line_chart,create_moving_average_chart,create_seasonal_decomposition_plot,create_time_series_comparison_chart,create_time_series_area_chart,create_correlation_matrix_heatmap,create_regression_analysis_visualization,create_residual_plot,create_distribution_comparison_chart,create_statistical_significance_visualization,create_choropleth_map,create_geospatial_scatter_map,create_location_density_heatmap,create_regional_comparison_map,create_node_link_graph,create_dependency_graph,create_relationship_network_graph,create_pca_visualization,create_tsne_plot,create_umap_plot,create_treemap,create_sunburst_chart,create_dendrogram,create_large_dataset_scatter_aggregation,create_large_dataset_density_visualization,create_scalable_heatmap,create_interactive_scatter_plot,create_interactive_time_series,create_hover_enabled_bar_chart,create_zoomable_heatmap,create_missing_value_heatmap,create_outlier_detection_plot,create_distribution_comparison_before_after


model=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")

tools=[ create_histogram,create_kde_plot,create_box_plot,create_violin_plot,create_frequency_bar_chart,create_pie_chart,create_scatter_plot,create_regression_plot,create_grouped_bar_chart,create_box_plot_by_category,create_violin_plot_by_category,create_categorical_comparison_chart,create_pair_plot,create_correlation_heatmap,create_bubble_chart,create_grouped_scatter_plot,create_stacked_bar_chart,create_cluster_heatmap,create_parallel_coordinates_plot,create_radar_chart,create_time_series_line_chart,create_moving_average_chart,create_seasonal_decomposition_plot,create_time_series_comparison_chart,create_time_series_area_chart,create_correlation_matrix_heatmap,create_regression_analysis_visualization,create_residual_plot,create_distribution_comparison_chart,create_statistical_significance_visualization,create_choropleth_map,create_geospatial_scatter_map,create_location_density_heatmap,create_regional_comparison_map,create_node_link_graph,create_dependency_graph,create_relationship_network_graph,create_pca_visualization,create_tsne_plot,create_umap_plot,create_treemap,create_sunburst_chart,create_dendrogram,create_large_dataset_scatter_aggregation,create_large_dataset_density_visualization,create_scalable_heatmap,create_interactive_scatter_plot,create_interactive_time_series,create_hover_enabled_bar_chart,create_zoomable_heatmap,create_missing_value_heatmap,create_outlier_detection_plot,create_distribution_comparison_before_after]

visualization_tools_node = ToolNode(tools=tools)


def Data_visualization( state : AgentState ) -> AgentState:
    
    tools_priority_list = state['tool_priority_list_3']

    llm_with_tools=model.bind_tools(tools)

    prompt = f"""
    
    You are a Data Visualization Agent in a Data Analytics system.

    Your responsibility is to execute data visualization tools in the correct order based on a provided execution list.

    INPUT FORMAT:
    You will receive a list of dictionaries. Each dictionary has the following structure:

    [
    {{
        "tool_name": {{
            "param1": value1,
            "param2": value2
        }}
    }},
    {{
        "another_tool": {{
            "paramA": valueA
        }}
    }}
    ]

    IMPORTANT RULES:

    1. The list represents a sequence of tools to be executed in PRIORITY ORDER.
    2. The FIRST element in the list is always the NEXT tool that must be executed.
    3. Each dictionary contains:
    - The KEY → name of the tool to call
    - The VALUE → dictionary containing the tool's input parameters.

    YOUR TASK:

    1. Always select the FIRST dictionary from the list.
    2. Extract:
    - the tool name (key)
    - the tool parameters (value dictionary)
    3. Call the corresponding tool using the extracted parameters.
    4. After the tool is executed, remove that dictionary from the list.
    5. Repeat the process with the updated list.

    STOP CONDITION:

    If the list becomes empty, STOP execution and return:

    "All data visualization tools have been executed successfully."

    IMPORTANT CONSTRAINTS:

    - Never skip tools.
    - Never change the execution order.
    - Only execute ONE tool at a time.
    - Always execute the tool at index 0 of the list.
    - Do not attempt to infer or modify tool parameters.
    - Do not fabricate new tools.

    OUTPUT BEHAVIOR:

    If the list is not empty:
    → Call the tool from the first dictionary.

    If the list is empty:
    → Return a completion message and stop further tool calls.

    here is the priority list of tools you should follow when executing the visualization process:

    ToolPriorityList: {tools_priority_list}
    """
    
    response=llm_with_tools.invoke(prompt)
    
    return {'messages': [response]}
    