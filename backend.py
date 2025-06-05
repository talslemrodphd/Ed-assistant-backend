from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import dropbox
import tempfile
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
DROPBOX_TOKEN = os.getenv("DROPBOX_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Setup Dropbox and Langchain
dbx = dropbox.Dropbox(DROPBOX_TOKEN)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and index Dropbox documents
def load_documents_from_dropbox():
    documents = []

    for entry in dbx.files_list_folder("", recursive=True).entries:
        if isinstance(entry, dropbox.files.FileMetadata):
            path = entry.path_display
            if path.endswith((".pdf", ".txt", ".docx")):
                _, temp_path = tempfile.mkstemp()
                with open(temp_path, "wb") as f:
                    metadata, res = dbx.files_download(path)
                    f.write(res.content)

                try:
                    if path.endswith(".pdf"):
                        loader = PyPDFLoader(temp_path)
                    elif path.endswith(".docx"):
                        loader = Docx2txtLoader(temp_path)
                    else:
                        loader = TextLoader(temp_path)
                    
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = path
                    documents.extend(docs)
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
                finally:
                    os.remove(temp_path)

    return documents

print("Loading documents from Dropbox...")
raw_docs = load_documents_from_dropbox()
print(f"Loaded {len(raw_docs)} documents.")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(raw_docs)

print("Creating vector store...")
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

class Question(BaseModel):
    question: str

@app.post("/")
async def ask_question(q: Question):
    print(f"Received question: {q.question}")
    result = qa_chain(q.question)

    sources = set()
    for doc in result.get("source_documents", []):
        sources.add(doc.metadata.get("source", "Unknown"))

    return {
        "answer": result.get("result", "No answer found."),
        "sources": "\n".join(sources)
