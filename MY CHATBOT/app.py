# app.py (production-ready)
import os
from pathlib import Path
import logging
from flask import Flask, request, jsonify, render_template

# LangChain and provider imports
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

app = Flask(__name__, static_folder="static", template_folder="templates")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration: read API key from environment
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    logger.warning("GOOGLE_API_KEY is not set. Set it in the environment before running.")

# Paths (relative to this file)
BASE_DIR = Path(__file__).resolve().parent
DOC_PATH = BASE_DIR / "Document for Model Final.txt"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"

qa_chain = None

def initialize_chatbot():
    """
    Load document, build or load FAISS embeddings, and initialize RetrievalQA chain.
    """
    global qa_chain
    try:
        logger.info("Initializing chatbot...")

        # Ensure provider env var is set for underlying library
        if API_KEY:
            os.environ["GOOGLE_API_KEY"] = API_KEY

        # Check document exists
        if not DOC_PATH.exists():
            raise FileNotFoundError(f"Document file not found at: {DOC_PATH.resolve()}. "
                                    "Place your source text file there or update DOC_PATH.")

        # Build or load vectorstore
        if FAISS_INDEX_DIR.exists():
            logger.info("Loading FAISS index from disk...")
            vectorstore = FAISS.load_local(str(FAISS_INDEX_DIR))
        else:
            logger.info(f"Loading document from: {DOC_PATH}")
            loader = TextLoader(str(DOC_PATH), encoding="latin-1")
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents(documents)
            logger.info(f"Document split into {len(chunks)} chunks.")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            logger.info("Saving FAISS index to disk...")
            vectorstore.save_local(str(FAISS_INDEX_DIR))

        # Initialize LLM and QA chain
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
        logger.info("✅ Chatbot initialized successfully.")

    except Exception as exc:
        logger.exception("❌ Error during chatbot initialization: %s", exc)
        qa_chain = None

# Flask routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    if qa_chain is None:
        return jsonify({"error": "The chatbot is not initialized. Please check the server logs."}), 500

    body = request.get_json(silent=True) or {}
    user_message = body.get("message")
    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    try:
        # Robust call to the chain to handle different LangChain versions
        result = qa_chain({"query": user_message})
        if isinstance(result, str):
            answer = result
        elif isinstance(result, dict):
            answer = result.get("result") or result.get("answer") or result.get("output_text") or str(result)
        else:
            answer = str(result)
        return jsonify({"reply": answer})
    except Exception as exc:
        logger.exception("Error during question answering: %s", exc)
        return jsonify({"error": "Failed to get a response from the model."}), 500

if __name__ == "__main__":
    initialize_chatbot()
    # For production use a WSGI server. Debug False for safety.
    app.run(host="0.0.0.0", port=5000, debug=False)