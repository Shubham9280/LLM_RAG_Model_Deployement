# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y \
    curl \
    sudo \
    neofetch \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://ollama.ai/install.sh | sh


# Make port 80 available to the world outside this container
EXPOSE 80

# Run main.py when the container launches
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["bash", "entrypoint.sh"]
