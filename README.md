# DVBS2X — DVB-S2/X TX↔RX Loopback Suite

Lightweight DVB-S2/X transmitter→receiver loopback test bench and utilities for development and experimentation.

Quickstart
1. Create a venv and install deps:
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
2. Run a loopback example:
   python tests/test_tx_rx_loopback.py --max-frames 1 --esn0-db 10
3. Run tests:
   pytest -q

Structure
- tx/, rx/, common/ — signal chain modules
- tests/ — integration/test driver (loopback)
- config/, data/ — matrices and sample input
- docs/, examples/ — usage and helpers

See docs/getting_started.md for more details.