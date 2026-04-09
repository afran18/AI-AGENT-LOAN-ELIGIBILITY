from typing import Literal
import operator
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END

# Import from your existing files
from config import llm
from state import AgentState
from tools import loan_agent_tools

# 1. Helper to create an agent node
def create_agent(llm, tools, system_prompt: str, name: str):
    """Encapsulates the agent logic and ensures it tags its name in the output."""
    llm_with_tools = llm.bind_tools(tools)
    
    def agent_node(state: AgentState):
        result = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + state["messages"])
        # We overwrite the name so the router knows who just spoke
        result.name = name
        return {"messages": [result]}
    
    return agent_node

# 2. System Prompts
POLICY_ANALYST_PROMPT = """You are the Bank's Senior Policy Analyst. 
Your goal is to provide a FINAL VERDICT on the loan.

MANDATORY WORKFLOW:
1. Immediately use 'policy_search' to find the specific rules for the loan type mentioned (e.g., 'Auto Loan').
2. Compare the Customer Advocate's summary against these rules.
3. Be strict. If a document like 'Utility Bill' is required by policy but not in the profile, flag it.
4. End your response with a clear 'VERDICT: APPROVED/REJECTED/PENDING' and a summary.

DO NOT ask for permission to continue. You are the final stage of the process."""

CUSTOMER_ADVOCATE_PROMPT = """You are the Customer Advocate.
Your job is to use 'get_customer_profile' to fetch the applicant's data using the current_customer_id.
Summarize the profile clearly so the Policy Analyst can evaluate it."""

# 3. Creating the Nodes
# We add names to the agents to make routing easier
policy_member = create_agent(llm, loan_agent_tools, POLICY_ANALYST_PROMPT, name="Policy_Analyst")
advocate_member = create_agent(llm, loan_agent_tools, CUSTOMER_ADVOCATE_PROMPT, name="Customer_Advocate")
tool_node = ToolNode(loan_agent_tools)

# 4. Routing Logic
def should_continue(state: AgentState) -> Literal["tools", "policy_analyst", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "tools"
    
    # If the last message was from the Advocate, ALWAYS go to Analyst
    if getattr(last_message, "name", None) == "Customer_Advocate":
        return "policy_analyst"

    # If the Analyst has finished their thought (and didn't call a tool), we end
    if getattr(last_message, "name", None) == "Policy_Analyst":
        return "__end__"
    
    return "__end__"

# 5. Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("customer_advocate", advocate_member)
workflow.add_node("policy_analyst", policy_member)
workflow.add_node("tools", tool_node)

# Entry Point
workflow.set_entry_point("customer_advocate")

# Conditional Edges
workflow.add_conditional_edges("customer_advocate", should_continue)
workflow.add_conditional_edges("policy_analyst", should_continue)

# Standard Edges: Tools always return to the 'router' or specific node
# In this design, tools will return to the advocate initially, 
# then the advocate will pass to analyst once data is ready.
workflow.add_edge("tools", "customer_advocate")

# 6. Compile
app = workflow.compile()

# 7. Execution Loop
def run_loan_assessment(customer_id: str):
    initial_input = {
        "messages": [
            HumanMessage(content=(
                f"Assess loan application for customer {customer_id}. "
                "1. Advocate retrieves profile. "
                "2. Analyst verifies rules. "
                "3. Provide final eligibility verdict."
            ))
        ],
        "current_customer_id": customer_id,
        "analysis_complete": False
    }

    print(f"\n{'='*60}")
    print(f"STARTING MULTI-AGENT ASSESSMENT: {customer_id}")
    print(f"{'='*60}")

    for event in app.stream(initial_input):
        for node, values in event.items():
            if "messages" in values:
                last_msg = values["messages"][-1]
                agent_name = getattr(last_msg, "name", node).upper()
                print(f"\n[NODE: {agent_name}]")
                print(last_msg.content if last_msg.content else f"Calling tools: {last_msg.tool_calls}")

    print(f"\n{'='*60}")
    print("ASSESSMENT COMPLETE")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Test with a potentially tricky case
    run_loan_assessment("P003")