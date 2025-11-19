#!/bin/bash

MODEL_LIST="models.txt"
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

    echo "Querying model..."
    python chat.py "$MODEL" "$INPUT_FILE"

    echo "Done with $MODEL"
    echo
done < "$MODEL_LIST"

echo "===================================="
echo "All models processed."
echo "===================================="
