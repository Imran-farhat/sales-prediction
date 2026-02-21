@echo off
set "PY313=C:\Users\ACER\AppData\Local\Programs\Python\Python313\python.exe"

echo [DEBUG] Checking for Python at: %PY313%

if exist "%PY313%" (
    echo [OK] Starting Stock Dashboard with Python 3.13...
    "%PY313%" app.py
) else (
    echo [ERROR] Python 3.13 was not found at the expected location.
    echo Please make sure Python 3.13 is installed.
)

echo.
echo Press any key to exit...
pause
