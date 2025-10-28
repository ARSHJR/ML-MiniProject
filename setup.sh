#!/bin/bash
# Bash setup script for Fast Food Nutrition ML Mini Project

echo "=== Fast Food Nutrition ML Mini Project Setup ==="
echo ""

# Create virtual environment
echo "[1/4] Creating virtual environment (.venv)..."
python3 -m venv .venv

if [ $? -ne 0 ]; then
    echo "Error creating virtual environment. Please ensure Python is installed."
    exit 1
fi

# Activate virtual environment
echo "[2/4] Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "[3/4] Upgrading pip..."
python -m pip install --upgrade pip --quiet

# Install dependencies
echo "[4/4] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "✓ Setup complete!"
echo ""
echo "To run the application:"
echo "  1. Activate the virtual environment:"
echo "     source .venv/bin/activate"
echo "  2. Run the Streamlit app:"
echo "     streamlit run app.py"
echo ""
