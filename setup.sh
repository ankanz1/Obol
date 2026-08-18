#!/bin/bash
# Setup script for Voice RAG

set -e

echo "🚀 Setting up Voice RAG for MSMARCO-XI..."

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install backend dependencies
echo "Installing backend dependencies..."
cd backend
pip install -r requirements.txt
cd ..

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Create .env from example if not exists
if [ ! -f ".env" ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your API keys!"
fi

# Create data directories
mkdir -p backend/data/{raw,chunks,embeddings}

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your Sarvam and Qdrant API keys"
echo "2. Run: python backend/scripts/build_index.py --full --recreate"
echo "3. Start backend: cd backend && python -m app.main"
echo "4. Start frontend: cd frontend && npm run dev"
echo "5. Open http://localhost:3000"