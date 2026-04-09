from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schema import FinalAssessmentResponse, AssessmentStep, LoanVerdict
from main import app as langgraph_app
import json
from langchain_core.messages import HumanMessage

app = FastAPI(title="Banking Loan Eligibility Agent API")

# Enable CORS so your React app (likely on port 3000 or 5173) can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your React URL
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/assess/{customer_id}", response_model=FinalAssessmentResponse)
async def assess_loan(customer_id: str):
    try:
        initial_input = {
            "messages": [HumanMessage(content=f"Assess loan for {customer_id}")],
            "current_customer_id": customer_id
        }

        steps = []
        final_verdict = None
        customer_name = "Unknown"

        # Stream the graph execution
        # Use sync 'stream' or async 'astream' based on your preference
        for event in langgraph_app.stream(initial_input):
            for node, values in event.items():
                if "messages" in values:
                    last_msg = values["messages"][-1]
                    agent_name = getattr(last_msg, "name", node)
                    
                    # 1. Capture Steps for the Frontend Timeline
                    steps.append(AssessmentStep(
                        agent=agent_name,
                        content=last_msg.content if last_msg.content else "Processing...",
                        is_tool_call=bool(last_msg.tool_calls)
                    ))

                    # 2. Extract Customer Name (from Advocate's tool output or summary)
                    if agent_name == "Customer_Advocate" and "applicant_name" in last_msg.content:
                        # Simple heuristic or parse from structured data if available
                        pass 

                    # 3. Capture the Final Pydantic Verdict
                    if agent_name == "Verdict_Writer":
                        verdict_data = json.loads(last_msg.content)
                        final_verdict = LoanVerdict(**verdict_data)

        if not final_verdict:
            raise HTTPException(status_code=500, detail="Agent failed to reach a verdict.")

        return FinalAssessmentResponse(
            customer_id=customer_id,
            customer_name=customer_name, # Optionally pull this from your CSV tool
            steps=steps,
            verdict=final_verdict
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)