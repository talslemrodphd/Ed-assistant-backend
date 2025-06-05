import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders import DropboxLoader

app = FastAPI()

# Allow CORS so your frontend can call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get Dropbox token from environment variable
DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")

if not DROPBOX_ACCESS_TOKEN:
    raise ValueError("Missing DROPBOX_ACCESS_TOKEN environment variable")

# Initialize DropboxLoader with folder path
loader = DropboxLoader(
    access_token=DROPBOX_ACCESS_TOKEN,
    folder_path="/Ed SPED Assistant"
)

@app.post("/")
async def answer_question(request: Request):
    data = await request.json()
    question = data.get("question", "")

    # For now, a simple placeholder response
    # Later, integrate your AI and document processing here
    answer = f"You asked: '{question}'. This is a placeholder answer."
    sources = "Dropbox folder: /Ed SPED Assistant"

    return {"answer": answer, "sources": sources}
