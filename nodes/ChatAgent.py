from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from tools.rag_tool import rag_chat_tool
from utils.state import AgentState
from langgraph.types import interrupt

model=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")

tools = [rag_chat_tool]

rag_tool_node = ToolNode(tools=tools)

llm_with_tools = model.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(content="""
You are an intelligent data analysis assistant specializing in answering questions 
about data analysis results stored in a knowledge base.

## Your Capabilities
You have access to a RAG (Retrieval-Augmented Generation) tool that searches through 
a vectorstore built from data analysis result files. Always use this tool before 
answering — do not rely on general knowledge or assumptions.

## How to Behave
- ALWAYS call the `rag_chat_tool` with the user's query before formulating a response.
- Base your answers strictly on the retrieved content. If the tool returns insufficient 
  information, say so clearly — do not hallucinate or fill in gaps.
- When the retrieved context is rich, synthesize it into a clear, concise answer.
- If the user's question is ambiguous, ask a brief clarifying question before retrieving.
- Maintain conversational context across turns — refer back to earlier questions 
  when relevant.

## Response Style
- Be precise and factual. This is analytical data — accuracy matters.
- Use bullet points or short paragraphs for clarity when listing findings.
- If a numeric result or statistic is mentioned, quote it exactly as retrieved.
- Keep responses focused; avoid unnecessary filler.

## Boundaries
- Only answer questions related to the data analysis results in the knowledge base.
- If asked something outside this scope, politely redirect the user.
- When the user types 'done', do not answer further — the system will proceed to 
  generate visualizations.
""")


def _extract_text(content) -> str:
    """Safely extract plain string from str or list[dict] Gemini response."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    return str(content).strip()



def chat_node(state: AgentState):
    # Build full message history
    messages = [SYSTEM_PROMPT]
    for turn in state.get("chat_history", []):
        messages.append(HumanMessage(content=turn["user"]))
        messages.append(AIMessage(content=turn["assistant"]))

    # Pause and wait for user input
    user_input = interrupt(
        "Ask me anything about the data analysis results "
        "(or type 'done' to proceed to visualizations):"
    )

    if user_input.strip().lower() == "done":
        return {
            "chat_active": False,
            "messages": [HumanMessage(content=user_input)],  # ← log it
        }

    messages.append(HumanMessage(content=user_input))

    # First LLM call — may produce a tool call
    response = llm_with_tools.invoke(messages)

    final_answer = ""

    if response.tool_calls:
        messages.append(response)  # AIMessage with tool_calls metadata

        # Execute each tool call and collect results
        for tool_call in response.tool_calls:
            tool_result = rag_chat_tool.invoke(tool_call)
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                )
            )

        # Second LLM call with tool results in context
        final_response = llm_with_tools.invoke(messages)
        final_answer = _extract_text(final_response.content)  # ← safe extraction
    else:
        final_answer = _extract_text(response.content)        # ← safe extraction

    if not final_answer:
        final_answer = "I couldn't generate a response based on the retrieved data."


    return {
        "chat_history": [{"user": user_input, "assistant": final_answer}],
        "chat_active": True,
        "messages": [
            HumanMessage(content=user_input),
            AIMessage(content=final_answer),      # ← this was missing
        ],
    }


def route_chat(state: AgentState) -> str:
    last = state["messages"][-1]

    # Tool call requested → RAG
    if getattr(last, "tool_calls", None):
        return "tools"

    # Still in interactive chat → loop back
    if state.get("chat_active", False):
        return "Chat_node"

    # Done chatting → move to visualization
    return "PriorityGenerator3_node"




