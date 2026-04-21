#!/bin/bash
echo "============================================"
echo "  CineAI - AI Movie Recommender Setup"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found!"
    echo "Please install Python 3.9+ from https://python.org"
    exit 1
fi

echo "[1/3] Installing required packages..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install packages."
    echo "Try: pip3 install --break-system-packages -r requirements.txt"
    exit 1
fi

echo ""
echo "[2/3] Training the AI recommendation model..."
python3 model/train_model.py
if [ $? -ne 0 ]; then
    echo "[ERROR] Model training failed."
    exit 1
fi

echo ""
echo "[3/3] Setup complete!"
echo ""
echo "============================================"
echo "  Run the app:  python3 app.py"
echo "  Then open:    http://localhost:5000"
echo "============================================"
