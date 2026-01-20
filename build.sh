#!/bin/bash
# Render build script

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir

echo "Creating directories..."
mkdir -p uploads logs outputs/plots

echo "Build complete!"
