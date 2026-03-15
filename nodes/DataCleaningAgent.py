from langchain_google_genai import ChatGoogleGenerativeAI
from utils.state import AgentState
from langgraph.prebuilt import ToolNode, tools_condition
from tools.data_cleaning_tools import handle_missing_values,detect_and_remove_duplicates,detect_outliers,fix_dtypes,standardize_data,transform_features

model=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
tools=[ handle_missing_values,detect_and_remove_duplicates,detect_outliers,fix_dtypes,standardize_data,transform_features]

cleaning_tools_node = ToolNode(tools=tools)

def Clean_data(state:AgentState)->AgentState:

    # report = state['report']
    # used_tools = state['used_tools']
    tools_priority_list_1 = state['tool_priority_list_1']

    llm_with_tools=model.bind_tools(tools)

    prompt = f"""

    You are a Data Cleaning Agent in a Data Analytics system.

    Your responsibility is to execute data cleaning tools in the correct order based on a provided execution list.

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

    "All data cleaning tools have been executed successfully."

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
    
    here is the priority list of tools you should follow when executing the cleaning process:
    
    ToolPriorityList: {tools_priority_list_1}

    """
    
    response=llm_with_tools.invoke(prompt)
    
    return {'messages': [response]}
    
    
    
    
def pop_executed_node_a(state : AgentState) -> AgentState:
    
    current_list = state['tool_priority_list_1']
    return {'tool_priority_list_1': current_list[1:]}

def pop_executed_node_b(state : AgentState) -> AgentState:
    
    current_list = state['tool_priority_list_2']
    return {'tool_priority_list_2': current_list[1:]}

def pop_executed_node_c(state : AgentState) -> AgentState:
    
    current_list = state['tool_priority_list_3']
    return {'tool_priority_list_3': current_list[1:]}
    