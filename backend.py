from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import dropbox
from langchain_community.document_loaders import DropboxLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

# Initialize FastAPI
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body model
class Question(BaseModel):
    question: str

# Load Dropbox token and OpenAI key
DROPBOX_TOKEN = os.getenv("DROPBOX_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load files from Dropbox App Folder
loader = DropboxLoader(access_token=DROPBOX_TOKEN)
documents = loader.load()

# Set up vector store
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(documents, embeddings)

# Set up the retrieval-based QA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

@app.post("/")
async def ask_question(q: Question):
    try:
        result = qa.run(q.question)
        return {"answer": result, "sources": "Dropbox"}
    except Exception as e:
        return {"answer": "Error occurred.", "sources": str(e)}
