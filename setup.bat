@echo off
echo ============================================
echo   CineAI - AI Movie Recommender Setup
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo Your Python version:
python --version
echo.

echo [1/3] Upgrading pip first...
python -m pip install --upgrade pip

echo.
echo [2/3] Installing required packages...
pip install flask pandas scikit-learn numpy
IF ERRORLEVEL 1 (
    echo.
    echo Trying alternative install method...
    pip install --only-binary=:all: flask pandas scikit-learn numpy
)

echo.
echo [3/3] Training the AI recommendation model...
python model/train_model.py
IF ERRORLEVEL 1 (
    echo [ERROR] Model training failed.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Setup complete!
echo   Run the app:  python app.py
echo   Then open:    http://localhost:5000
echo ============================================
pause
