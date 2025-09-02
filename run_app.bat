@echo off


REM Activate the virtual environment
call venv\Scripts\activate

REM Start the FastAPI application
uvicorn main:app --host 0.0.0.0 --port 8000

REM Keep the window open after it runs
pause
