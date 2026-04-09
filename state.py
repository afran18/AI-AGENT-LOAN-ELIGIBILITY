import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # The list of messages in the conversation
    # Annotated with operator.add means new messages are appended to the history
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # Track which customer we are currently analyzing
    current_customer_id: str
    # A flag to signal if the analysis is complete
    analysis_complete: bool