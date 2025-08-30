@echo off

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Run the Python script
python -m comms.zmq_worker
