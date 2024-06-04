#!/bin/bash

# Start the Ollama service
nohup ollama serve &

# Wait for the Ollama service to start
sleep 10

# Pull the model
ollama pull phi3

# Run the Flask application
python main.py
