from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Allow your frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to just your frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question", "")
    
    # Just for testing: return a fake answer
    return JSONResponse({
        "answer": f"You asked: '{question}'. This is a placeholder answer.",
        "sources": "Dropbox placeholder"
    })
