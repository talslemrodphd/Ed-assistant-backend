import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import DropboxLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Read environment variables
DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
DROPBOX_FOLDER_PATH = os.getenv("DROPBOX_FOLDER_PATH", "/Ed SPED Assistant")

# Debug logging
print("DROPBOX_ACCESS_TOKEN is set:", bool(DROPBOX_ACCESS_TOKEN))
print("DROPBOX_FOLDER_PATH:", DROPBOX_FOLDER_PATH)

class QuestionRequest(BaseModel):
    question: str

@app.post("/")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question")
    if not question:
        return {"answer": "Please provide a question."}

    if not DROPBOX_ACCESS_TOKEN:
        return {"answer": "Error: DROPBOX_ACCESS_TOKEN is missing."}

    try:
        # Initialize loader with folder_path and access token
        loader = DropboxLoader.from_params({
            "access_token": DROPBOX_ACCESS_TOKEN,
            "folder_path": DROPBOX_FOLDER_PATH
        })
        docs = loader.load()

        # Vector store setup
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(docs, embeddings)
        retriever = vector_store.as_retriever()

        # Retrieval QA
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        result = qa.run(question)

        return {
            "answer": result,
            "sources": f"Dropbox folder: {DROPBOX_FOLDER_PATH}"
        }

    except Exception as e:
        return {
            "answer": f"Error processing your request: {str(e)}",
            "sources": "None"
        }
