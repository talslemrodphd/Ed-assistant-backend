from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import DropboxLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import os

app = FastAPI()

# Allow frontend to connect (update this if your frontend URL changes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can change "*" to your specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.post("/")
async def ask_question(q: Question):
    # Load environment variables
    dropbox_token = os.getenv("DROPBOX_ACCESS_TOKEN")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Load documents from Dropbox folder
    loader = DropboxLoader(
        access_token=dropbox_token,
        folder_path="/Ed SPED Assistant"  # <- your folder name in Dropbox
    )
    documents = loader.load()

    # Split into smaller chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Embed and store in vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(docs, embeddings)

    # Search for relevant documents
    relevant_docs = db.similarity_search(q.question)

    # Run question-answering chain
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=relevant_docs, question=q.question)

    # Return answer and sources (simple source display)
    sources = "\n".join(set(doc.metadata.get("source", "Unknown") for doc in relevant_docs))
    return {"answer": answer, "sources": sources}
