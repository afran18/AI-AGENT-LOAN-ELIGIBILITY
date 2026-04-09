import json
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END


# Import shared components
from config import llm
from state import AgentState
from tools import loan_agent_tools
from schema import LoanVerdict

# --- 1. Node Definitions ---

def advocate_node(state: AgentState):
    """Fetches customer data and summarizes the profile."""
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(loan_agent_tools)
    prompt = """You are the Customer Advocate. 
    Use 'get_customer_profile' to fetch the applicant's data.
    Summarize their financial health clearly for the Policy Analyst."""
    
    result = llm_with_tools.invoke([SystemMessage(content=prompt)] + state["messages"])
    result.name = "Customer_Advocate"
    return {"messages": [result]}

def policy_analyst_node(state: AgentState):
    """Queries the policy documents and performs the comparison logic."""
    llm_with_tools = llm.bind_tools(loan_agent_tools)
    
    # We provide the Analyst with a strict reasoning prompt
    prompt = """You are the Senior Policy Analyst. 
    1. Use 'policy_search' to find the rules for the requested loan type.
    2. Compare the Advocate's summary against these rules.
    3. If you have all the info, provide a detailed final analysis.
    
    IMPORTANT: Once you have the rules and the data, provide your final response."""
    
    result = llm_with_tools.invoke([SystemMessage(content=prompt)] + state["messages"])
    result.name = "Policy_Analyst"
    return {"messages": [result]}

def verdict_node(state: AgentState):
    """The final 'structured' node that converts reasoning into Pydantic JSON."""
    structured_llm = llm.with_structured_output(LoanVerdict)
    
    prompt = "Review the conversation and output the final loan decision in structured JSON format."
    
    # This invokes the LLM to fill our Pydantic LoanVerdict model
    result = structured_llm.invoke([SystemMessage(content=prompt)] + state["messages"])
    
    # We save the Pydantic model as a JSON string so FastAPI can parse it easily
    res_msg = HumanMessage(content=result.model_dump_json(), name="Verdict_Writer")
    return {"messages": [res_msg]}

# --- 2. Routing Logic ---

def should_continue(state: AgentState) -> Literal["tools", "policy_analyst", "verdict_writer", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    
    # 1. Check if the last message is an AI Message AND has tool calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    
    # 2. Use getattr with a default of None to safely check the name
    agent_name = getattr(last_message, "name", None)

    if agent_name == "Customer_Advocate":
        return "policy_analyst"
    
    if agent_name == "Policy_Analyst":
        return "verdict_writer"
    
    return "__end__"

def route_tools(state: AgentState):
    """Routes tool results back to the agent that called them."""
    messages = state["messages"]
    # We look back through history to find which AI agent initiated the call
    for msg in reversed(messages):
        # We check if it's an AIMessage and has a name attribute
        msg_name = getattr(msg, "name", None)
        if msg_name in ["Customer_Advocate", "Policy_Analyst"]:
            return msg_name.lower()
    
    return "customer_advocate"

# --- 3. Graph Construction ---

workflow = StateGraph(AgentState)

workflow.add_node("customer_advocate", advocate_node)
workflow.add_node("policy_analyst", policy_analyst_node)
workflow.add_node("verdict_writer", verdict_node)
workflow.add_node("tools", ToolNode(loan_agent_tools))

workflow.set_entry_point("customer_advocate")

workflow.add_conditional_edges("customer_advocate", should_continue)
workflow.add_conditional_edges("policy_analyst", should_continue)
workflow.add_conditional_edges("tools", route_tools)
workflow.add_edge("verdict_writer", END)

app = workflow.compile()

# --- 4. Execution Loop (CLI Test) ---

def run_loan_assessment(customer_id: str):
    initial_input = {
        "messages": [HumanMessage(content=f"Full assessment for {customer_id}")],
        "current_customer_id": customer_id
    }

    print(f"\nProcessing: {customer_id}")
    final_output = None

    for event in app.stream(initial_input):
        for node, values in event.items():
            if "messages" in values:
                last_msg = values["messages"][-1]
                name = getattr(last_msg, "name", node).upper()
                
                # If it's the final verdict node, parse the JSON for the CLI display
                if name == "VERDICT_WRITER":
                    final_output = json.loads(last_msg.content)
                    print(f"\n[{name}] -> DECISION: {final_output['decision']}")
                else:
                    print(f"\n[{name}]")
                    print(last_msg.content[:200] + "..." if last_msg.content else "Calling tools...")

    return final_output

if __name__ == "__main__":
    run_loan_assessment("P003")