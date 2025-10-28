# MY-CHATBOT (FinTech) — Deployment Guide

This repository runs a Flask app that builds a FAISS vector index of a provided text document and serves a retrieval-augmented QA endpoint using Google Generative AI (Gemini) via LangChain.

IMPORTANT: Do NOT commit your API keys to the repo. Use environment variables or a secrets manager.

## Files added in fix/app-deploy
- `MY CHATBOT/app.py` — Production-ready Flask app (reads key from env, persists FAISS).
- `MY CHATBOT/student_rbi_chatbot_final_code.py` — Cleaned local experiment script.
- `MY CHATBOT/requirements.txt` — Python deps to install (pin exact versions before deploy).
- `MY CHATBOT/.gitignore` — Ignore secrets, local files, FAISS index.
- `Dockerfile` — Containerize the app.
- `README.md` — This file.

## Quick local run (non-Docker)
1. Create a Python virtual environment:
   python -m venv venv
   source venv/bin/activate

2. Install dependencies:
   pip install -r "MY CHATBOT/requirements.txt"

3. Place your document:
   - Put your text file named `Document for Model Final.txt` into the `MY CHATBOT` folder.
   - Alternatively, change the DOC_PATH in `app.py` to match your file.

4. Set your Gemini API key:
   export GOOGLE_API_KEY="your_real_gemini_api_key_here"   # Linux / macOS
   setx GOOGLE_API_KEY "your_real_gemini_api_key_here"     # Windows (or set env in PowerShell)

5. Run locally:
   cd "MY CHATBOT"
   python app.py

6. POST to /chat:
   curl -X POST -H "Content-Type: application/json" \
     -d '{"message":"Summarize the document"}' \
     http://localhost:5000/chat

## Docker run
1. Build the image (from repo root):
   docker build -t my-chatbot:latest .

2. Run the container (pass key via env):
   docker run -e GOOGLE_API_KEY="your_real_gemini_api_key_here" -p 5000:5000 my-chatbot:latest

## Notes / Recommendations
- Pin exact LangChain and provider package versions that you tested.
- Persist the FAISS index (we save to `faiss_index/` by default). Back up the index if you rebuild containers.
- Use a WSGI like Gunicorn (Dockerfile already uses Gunicorn).
- For production, add authentication or at least IP-based protection on /chat.
- Use a managed secrets store (AWS Secrets Manager, GCP Secret Manager, or GitHub Secrets for CI/CD).
- Monitor costs and quota for Gemini usage.

## Next steps I can take for you
- Push the remaining prepared files to the `fix/app-deploy` branch and open a PR against `main` with a descriptive message.
- Pin exact package versions after you confirm the working environment and update `requirements.txt`.
- Add a GitHub Actions workflow to build and publish the Docker image automatically.
- Help deploy to a target platform (Cloud Run, ECS, DigitalOcean App Platform), including required env var configuration and scaling considerations.
