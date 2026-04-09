import pandas as pd
from langchain.tools import tool
from langchain_chroma import Chroma
from config import embedding_model

# 1. Setup retriever for the PDF
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(search_kwargs={"k" : 3})

@tool
def policy_search(query: str):
    """Search the bank's loan policy documents for specific rules,
    eligibility criteria and required documentation."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

@tool
def get_customer_profile(customer_id: str):
    """Retrieve detailed customer profile data from the CSV database including credit score, income and existing debt."""
    try:
        df = pd.read_csv("data/CustomerProfiles_LoanEligibility.csv")
        profile = df[df['customer_id'] == customer_id]
        if profile.empty:
            return f"No customer found with ID: {customer_id}"
        return profile.to_dict(orient='records')[0]
    except Exception as e:
        return f"Error reading customer data: {e}"
    
# List of tools to be used by the agents
loan_agent_tools = [policy_search, get_customer_profile]