#!/bin/bash

# Quality Benchmarking Agent Setup Script

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
REQUIRED_VERSION="3.8"

# Compare Python versions
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$PYTHON_VERSION" ]; then
    echo "❌ Python version must be >= 3.8. Current version: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Download NLP resources
echo "🧠 Downloading NLP resources..."
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt
python -m nltk.downloader wordnet

# Verify installations
echo "🔍 Verifying installations..."
python -c "
import spacy
import nltk

print('SpaCy model check:', 'Passed' if spacy.load('en_core_web_sm') else 'Failed')
print('NLTK Punkt check:', 'Passed' if nltk.data.find('tokenizers/punkt') else 'Failed')
print('NLTK WordNet check:', 'Passed' if nltk.data.find('corpora/wordnet') else 'Failed')
"

# Run the application
echo "🚀 Starting Quality Benchmarking Agent..."
streamlit run analyzer.py
