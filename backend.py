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

# Allow CORS from any origin (for frontend to communicate with backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

# Load environment variables
DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
DROPBOX_FOLDER_PATH = "/Ed SPED Assistant"  # Adjust folder path in Dropbox here

# Initialize document loader for Dropbox folder
loader = DropboxLoader(
    access_token=DROPBOX_ACCESS_TOKEN,
    folder_path=DROPBOX_FOLDER_PATH
)

# Load documents from Dropbox
documents = loader.load()

# Create vector embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Set up retrieval-based QA chain with OpenAI chat model
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o-mini"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

@app.post("/")
async def answer_question(request: Request, question_request: QuestionRequest):
    question = question_request.question
    result = qa_chain.run(question)
    
    # For sources, we could customize this if your chain supports source documents
    sources = "Sources are not yet implemented."

    return {
        "answer": result,
        "sources": sources,
    }
