#!/bin/bash

# Script to copy Hugging Face checkpoint directory while removing optimizer-related files
# Usage: ./ckpt2model.sh <source_dir> <destination_dir>

# Check for correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_dir> <destination_dir>"
    exit 1
fi

SOURCE_DIR=$1
DEST_DIR=$2

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist."
    exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Copy only model-related files while excluding optimizer files
rsync -av --include='*/' \
    --exclude='optimizer.pt' \
    --exclude='rng_state.pth' \
    --exclude='scheduler.pt' \
    --exclude='trainer_state.json' \
    "$SOURCE_DIR"/ "$DEST_DIR"

# Output success message
echo "Hugging Face checkpoint has been copied to '$DEST_DIR' without optimizer-related files."

