@echo off
REM NMR Processing Enhanced UI Launcher
echo Activating matlab312 environment...
call conda activate matlab312
echo Starting Enhanced NMR Processing UI...
python ui_nmr_processing_enhanced.py
pause
