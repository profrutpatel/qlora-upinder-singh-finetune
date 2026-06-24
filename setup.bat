@echo off
:: ============================================================
:: QLoRA Fine-tuning Setup Script (Windows)
:: Run this ONCE after installing Python
:: ============================================================

echo.
echo  ====================================================
echo   QLoRA Fine-tuning Setup
echo  ====================================================
echo.

:: Check Python
python --version 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    echo.
    echo Please install Python 3.10 or 3.11 from:
    echo   https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo.
echo [2/4] Installing PyTorch with CUDA support...
echo (This downloads ~2GB, please wait...)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [3/4] Installing fine-tuning dependencies...
pip install -r requirements.txt

echo.
echo [4/4] Verifying CUDA availability...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo  ====================================================
echo   Setup complete! You can now run:
echo.
echo   venv\Scripts\activate
echo   python finetune.py
echo  ====================================================
echo.
pause
