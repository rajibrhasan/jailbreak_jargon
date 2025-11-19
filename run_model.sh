#!/bin/bash

MODEL_LIST="models.txt"
PORT=8000
INPUT_FILE="base.json"

if [ ! -f "$MODEL_LIST" ]; then
    echo "Error: models.txt not found!"
    exit 1
fi

while IFS= read -r MODEL; do
    if [ -z "$MODEL" ]; then
        continue
    fi

    SAFE_NAME=${MODEL//\//_}
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

    echo "===================================="
    echo "Running model: $MODEL"
    echo "Input file:   $INPUT_FILE"
    echo "===================================="

    # Start vLLM server
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port $PORT &

    SERVER_PID=$!
    echo "Server PID = $SERVER_PID"

    echo "Waiting for server..."
    sleep 8

    echo "Querying model..."
    python chat.py "$MODEL" "$INPUT_FILE" "$PORT"

    echo "Stopping vLLM server..."
    kill $SERVER_PID
    sleep 3

    echo "Done with $MODEL"
    echo
done < "$MODEL_LIST"

echo "===================================="
echo "All models processed."
echo "===================================="
