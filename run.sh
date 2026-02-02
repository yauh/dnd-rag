#!/bin/bash
# RAG System Convenience Script

cd /Volumes/corsair4tb/Users/stephan/Documents/code/RAG

# Activate venv
source venv/bin/activate

# Set environment variables
export QDRANT_URL="http://localhost:6333"
export LM_STUDIO_URL="http://localhost:1234/v1"
export TRANSFORMERS_CACHE="/Volumes/corsair4tb/Users/stephan/docker/rag_cache"
export HF_HOME="/Volumes/corsair4tb/Users/stephan/docker/rag_cache"

# Check if Qdrant is running
if ! docker ps | grep -q rag-qdrant; then
    echo "Starting Qdrant..."
    docker-compose up -d qdrant
fi

# Run the command
python scripts/"$@"
