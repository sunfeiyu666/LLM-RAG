@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "..\LLM-RAG-local\tools\start_local.ps1" -Mode Check
if errorlevel 1 goto done
"..\LLM-RAG-local\.venv\Scripts\python.exe" -X utf8 src\run_t3.py --stage all
if errorlevel 1 goto done
"..\LLM-RAG-local\.venv\Scripts\python.exe" -X utf8 tools\run_t3_challenges.py
if errorlevel 1 goto done
"..\LLM-RAG-local\.venv\Scripts\python.exe" -X utf8 tools\build_t3_cases.py
if errorlevel 1 goto done
"..\LLM-RAG-local\.venv\Scripts\python.exe" -X utf8 tools\summarize_t3.py
:done
pause
