#!/bin/bash
# Render build script

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p uploads logs outputs/plots
