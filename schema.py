from pydantic import BaseModel, Field
from typing import List, Optional


class AssessmentStep(BaseModel):
    agent: str = Field(description="The name of the agent (Advocate or Analyst)")
    content: str = Field(description="The text content or summary from the agent")
    is_tool_call: bool = Field(default=False)

class LoanVerdict(BaseModel):
    decision: str = Field(description="Final decision APPROVED, REJECTED or PENDING")
    rationale: str = Field(description="A concise explanation of the decision based on policy")
    missing_documents: List[str] = Field(default_factory=list, description="Any docs identified as missing")

class FinalAssessmentResponse(BaseModel):
    customer_id: str
    customer_name: str
    steps: List[AssessmentStep]
    verdict: LoanVerdict