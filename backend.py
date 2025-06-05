import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.document_loaders import DropboxLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

app = FastAPI()

# CORS setup to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Read environment variables
DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
DROPBOX_FOLDER_PATH = os.getenv("DROPBOX_FOLDER_PATH", "/Ed SPED Assistant")  # default folder path

# Debug prints - will appear in Render logs
print("DROPBOX_ACCESS_TOKEN:", DROPBOX_ACCESS_TOKEN)
print("DROPBOX_FOLDER_PATH:", DROPBOX_FOLDER_PATH)

# Define request schema
class QuestionRequest(BaseModel):
    question: str

@app.post("/")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question")
    if not question:
        return {"answer": "No question provided."}

    # Check environment variable presence
    if not DROPBOX_ACCESS_TOKEN:
        return {"answer": "Error: DROPBOX_ACCESS_TOKEN is not set."}

    try:
        # Load documents from Dropbox folder
        loader = DropboxLoader(
            access_token=DROPBOX_ACCESS_TOKEN,
            folder_path=DROPBOX_FOLDER_PATH,
        )
        docs = loader.load()

        # Create vector store and retriever
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(docs, embeddings)
        retriever = vector_store.as_retriever()

        # Create the QA chain with chat model
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        # Run the chain with the question
        result = qa_chain.run(question)

        # Return the answer (you can extend to return sources as well)
        return {"answer": result, "sources": "Dropbox folder: " + DROPBOX_FOLDER_PATH}

    except Exception as e:
        return {"answer": f"Error processing your request: {str(e)}"}
