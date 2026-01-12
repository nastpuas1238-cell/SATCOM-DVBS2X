# SatCom Python Project

This repository is a scaffold for implementing a satellite baseband processing chain in Python. It provides module stubs for each component (from the planning spreadsheet) so you can implement algorithmic details incrementally.

Quick start (Windows PowerShell):

```powershell
# create and activate a virtual environment
python -m venv .venv; .\.venv\Scripts\Activate.ps1
# install requirements
pip install -r requirements.txt
# run demo
python scripts\run_demo.py
```

What's included:
- `satcom/` - Python package with module stubs for each processing block.
- `scripts/run_demo.py` - simple runner showing how to wire stubs.
- `tests/` - basic tests to verify imports and stubs.

Next steps:
- Implement algorithms in each module (see module docstrings).
- Replace or add dependencies for performance (NumPy, SciPy, C extensions, or bindings to Liquid-DSP/GNU Radio).
- Add unit tests per module as you implement functionality.
