#!/bin/bash

echo "🏗️ Running build script..."

# Create models folder
mkdir -p models-gdrive

# Run Python script to download models
python download_models.py

echo "✅ Model files downloaded during build!"
