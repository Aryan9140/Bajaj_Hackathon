from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI()

class QueryRequest(BaseModel):
    data: list

class QueryResponse(BaseModel):
    is_success: bool
    user_id: str
    email: str
    roll_number: str
    numbers: list
    alphabets: list
    highest_lowercase_alphabet: list

@app.get("/")
async def root():
    return {"message": "Server is running"}

@app.post("/hackrx/run")
async def process_data(request: QueryRequest):
    try:
        data = request.data
        
        # Your logic here
        numbers = [item for item in data if item.isdigit()]
        alphabets = [item for item in data if item.isalpha()]
        lowercase_alphabets = [item for item in alphabets if item.islower()]
        highest_lowercase = [max(lowercase_alphabets)] if lowercase_alphabets else []
        
        return QueryResponse(
            is_success=True,
            user_id="patel",
            email="aryanpatel77462@gmail.com",
            roll_number="1047",
            numbers=numbers,
            alphabets=alphabets,
            highest_lowercase_alphabet=highest_lowercase
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Keep this for local testing, Render will ignore it
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)