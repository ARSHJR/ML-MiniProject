# PowerShell setup script for Fast Food Nutrition ML Mini Project

Write-Host "=== Fast Food Nutrition ML Mini Project Setup ===" -ForegroundColor Cyan
Write-Host ""

# Create virtual environment
Write-Host "[1/4] Creating virtual environment (.venv)..." -ForegroundColor Yellow
python -m venv .venv

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error creating virtual environment. Please ensure Python is installed." -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "[2/4] Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "[3/4] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# Install dependencies
Write-Host "[4/4] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "✓ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To run the application:" -ForegroundColor Cyan
Write-Host "  1. Activate the virtual environment:" -ForegroundColor White
Write-Host "     .\.venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "  2. Run the Streamlit app:" -ForegroundColor White
Write-Host "     streamlit run app.py" -ForegroundColor Gray
Write-Host ""
