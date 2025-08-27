# app.py (NEW AND IMPROVED CODE)
import os
from flask import Flask, request, jsonify, render_template

# --- Import All Necessary Libraries from Your Colab File ---
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

app = Flask(__name__)

# --- Configuration ---
# PASTE YOUR GEMINI API KEY HERE
API_KEY = "AIzaSyBEfUPhyvmWWKNbUaPafu3djJ6-K0rZvCw"

# Global variable to hold the question-answering chain
qa_chain = None

def initialize_chatbot():
    """
    This function loads the document, creates the vector store,
    and initializes the question-answering chain.
    """
    global qa_chain
    try:
        print("Initializing chatbot...")

        # 1. Set the API Key
        os.environ["GOOGLE_API_KEY"] = API_KEY

        # 2. Define the path to your document
        # This path assumes the file is in the same main folder as app.py
        file_path = "Document for Model Final.txt"

        # 3. Load the document
        print(f"Loading document from: {file_path}")
        loader = TextLoader(file_path, encoding='latin-1')
        documents = loader.load()
        print("Document loaded successfully.")

        # 4. Split the document into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        print(f"Document split into {len(chunks)} chunks.")

        # 5. Create embeddings and the vector store (FAISS)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("Vector store created successfully.")

        # 6. Initialize the Language Model and QA Chain
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

        print("✅ Chatbot initialized successfully!")

    except Exception as e:
        print(f"❌ Error during chatbot initialization: {e}")

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if not qa_chain:
        return jsonify({'error': 'The chatbot is not initialized. Please check the server logs.'}), 500

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided.'}), 400

    try:
        # Use the QA chain to get an answer from the document
        answer = qa_chain.run(user_message)
        return jsonify({'reply': answer})
    except Exception as e:
        print(f"Error during question answering: {e}")
        return jsonify({'error': 'Failed to get a response from the model.'}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # Initialize the chatbot when the server starts
    initialize_chatbot()
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)