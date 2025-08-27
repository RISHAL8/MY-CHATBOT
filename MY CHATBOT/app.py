# app.py
import os
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai

# --- Configuration ---
# To keep your API key secure, set it as an environment variable
# instead of hardcoding it. For now, you can paste it here.
# For deployment, we will use environment variables.
API_KEY = "AIzaSyBEfUPhyvmWWKNbUaPafu3djJ6-K0rZvCw"

app = Flask(__name__)

# --- Initialize the Gemini Model ---
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    print("Gemini model initialized successfully!")
except Exception as e:
    print(f"Error initializing Gemini model: {e}")
    model = None

# --- Define Routes ---
@app.route('/')
def index():
    # This will serve the HTML page we will create next
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if not model:
        return jsonify({'error': 'The generative model is not initialized.'}), 500

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided.'}), 400

    try:
        # Send message to the Gemini API
        response = model.generate_content(user_message)
        # Return the bot's reply
        return jsonify({'reply': response.text})
    except Exception as e:
        print(f"Error during API call: {e}")
        return jsonify({'error': 'Failed to get a response from the model.'}), 500

# --- Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)