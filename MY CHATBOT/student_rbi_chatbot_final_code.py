"""
Cleaned and portable version of the original Colab notebook.
- Remove notebook !pip calls.
- Read API key from environment.
- Use relative or configurable paths for the document.
This script is for local experimentation (not required by the Flask app).
"""

import os
from pathlib import Path
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Configuration
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    os.environ["GOOGLE_API_KEY"] = API_KEY
else:
    print("Warning: GOOGLE_API_KEY not set. Set it in the environment before running this script.")

# Document path (update as needed)
BASE_DIR = Path(__file__).resolve().parent
DOC_PATH = BASE_DIR / "Document for Model Final.txt"

def build_qa_chain(doc_path: Path):
    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found at: {doc_path}")
    loader = TextLoader(str(doc_path), encoding="latin-1")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa_chain

if __name__ == "__main__":
    qa = build_qa_chain(DOC_PATH)
    # Example questions
    qs = [
        "To which entities do these Digital Lending Directions apply?",
        "What is the maximum deposit insurance limit?"
    ]
    for q in qs:
        print("Question:", q)
        try:
            # Use .run or direct call depending on the chain; try both robustly
            try:
                res = qa.run(q)
                print("Answer:", res)
            except Exception:
                out = qa({"query": q})
                if isinstance(out, dict):
                    print("Answer:", out.get("result") or out.get("answer") or out.get("output_text") or out)
                else:
                    print("Answer:", out)
        except Exception as e:
            print("Error while answering:", e)
