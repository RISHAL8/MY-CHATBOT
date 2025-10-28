# Use a slim Python base
FROM python:3.11-slim

# Create app directory
WORKDIR /app

# Copy only the app folder (adjust if repo layout differs)
COPY "MY CHATBOT" /app

WORKDIR /app

# Install system deps (if any) then Python deps
RUN apt-get update && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY "MY CHATBOT/requirements.txt" /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Expose port
EXPOSE 5000

# Start the app with Gunicorn (production)
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
