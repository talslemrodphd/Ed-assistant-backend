import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.document_loaders import DropboxLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

app = FastAPI()

# Load environment variables
DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate environment variables
if not DROPBOX_ACCESS_TOKEN:
    raise ValueError("Missing DROPBOX_ACCESS_TOKEN environment variable")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY environment variable")

# Load documents from Dropbox folder
loader = DropboxLoader(
    access_token=DROPBOX_ACCESS_TOKEN,
    folder_path="/Ed SPED Assistant"  # Your Dropbox folder name
)
documents = loader.load()

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(documents, embeddings)

# Initialize retriever and QA chain
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

class Query(BaseModel):
    question: str

@app.post("/")
async def answer_question(query: Query):
    question = query.question
    if not question:
        return {"answer": "Please provide a question."}

    result = qa_chain.run(question)

    # Optionally include sources if available
    sources = ""
    if hasattr(result, "source_documents"):
        source_docs = result.source_documents
        sources = "\n".join(set([doc.metadata.get("source", "Unknown source") for doc in source_docs]))

    return {
        "answer": result,
        "sources": sources or "No sources found."
