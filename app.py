from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from schema import FinalAssessmentResponse, AssessmentStep, LoanVerdict
from main import app as langgraph_app
import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import pandas as pd
from io import StringIO
from tools import set_customer_data

app = FastAPI(title="Banking Loan Eligibility Agent API")

# Global variable to store uploaded customer data
customer_data_df = None

# Enable CORS so your React app (likely on port 3000 or 5173) can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your React URL
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a customer CSV file to be used for assessments."""
    try:
        global customer_data_df
        
        # Read the uploaded file
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        # Store in global variable in app.py
        customer_data_df = df
        
        # Also pass to tools.py
        set_customer_data(df)
        
        return {
            "status": "success",
            "message": f"CSV uploaded successfully with {len(df)} customer records",
            "total_customers": len(df),
            "available_customers": df['customer_id'].tolist() if 'customer_id' in df.columns else []
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error uploading CSV: {str(e)}")

@app.get("/upload-status")
async def upload_status():
    """Check if a CSV has been uploaded and list available customers."""
    try:
        if customer_data_df is None:
            return {
                "status": "no_data",
                "message": "No customer data uploaded yet",
                "total_customers": 0,
                "available_customers": []
            }
        
        return {
            "status": "ready",
            "message": f"CSV loaded successfully",
            "total_customers": len(customer_data_df),
            "available_customers": customer_data_df['customer_id'].tolist() if 'customer_id' in customer_data_df.columns else []
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error checking upload status: {str(e)}")

@app.get("/assess/{customer_id}", response_model=FinalAssessmentResponse)
async def assess_loan(customer_id: str):
    """Assess loan eligibility for a customer. Upload CSV first using /upload-csv endpoint."""
    try:
        global customer_data_df
        
        # If no data uploaded, try to load default CSV as fallback
        if customer_data_df is None:
            try:
                customer_data_df = pd.read_csv("data/CustomerProfiles_LoanEligibility.csv")
                set_customer_data(customer_data_df)
            except:
                raise HTTPException(status_code=400, detail="No customer data uploaded. Please upload a CSV file using /upload-csv endpoint.")
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
                        is_tool_call=isinstance(last_msg, AIMessage) and bool(last_msg.tool_calls)
                    ))

                    # 2. Extract Customer Name from tool result
                    if agent_name == "get_customer_profile":
                        try:
                            profile_data = json.loads(last_msg.content)
                            customer_name = profile_data.get("applicant_name", "Unknown")
                        except (json.JSONDecodeError, AttributeError):
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