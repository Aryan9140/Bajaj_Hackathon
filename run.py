from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Union
import uvicorn
import os

app = FastAPI()

# Original simple endpoint model
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

# Document processing model
class DocumentRequest(BaseModel):
    documents: str
    questions: List[str]

class DocumentResponse(BaseModel):
    answers: List[str]

@app.get("/")
async def root():
    return {"message": "Server is running"}

@app.post("/hackrx/run")
async def process_data(request: Request):
    try:
        request_data = await request.json()
        
        # Check if it's simple data processing
        if "data" in request_data:
            data = request_data["data"]
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
        
        # Check if it's document processing
        elif "documents" in request_data and "questions" in request_data:
            # For now, return placeholder responses
            # You'll need to implement actual document processing
            questions = request_data["questions"]
            placeholder_answers = [
                "This feature requires document processing implementation." 
                for _ in questions
            ]
            
            return DocumentResponse(answers=placeholder_answers)
        
        else:
            raise HTTPException(status_code=400, detail="Invalid request format")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)