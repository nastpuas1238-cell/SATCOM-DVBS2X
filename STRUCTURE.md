## Project Structure

```
DVBS2x/
├── tx/                          # Transmitter implementation
│   ├── run_dvbs2.py            # Main transmitter script
│   ├── BB_Frame.py             # Baseband frame generation
│   ├── stream_adaptation.py    # Rate adaptation
│   ├── bch_encoding.py         # BCH encoding
│   ├── ldpc_Encoding.py        # LDPC encoding
│   ├── pl_header.py            # PL header creation
│   ├── bbframe_report.py       # Reporting utility
│   ├── testing.py              # Transmitter tests
│   └── plot_qpsk_fft.py        # Visualization
│
├── rx/                          # Receiver implementation
│   ├── receiver_Chain.py       # Main receiver chain
│   ├── pl_descrambler.py       # PL frame descrambling
│   ├── pilot_removal_rx.py     # Pilot extraction
│   ├── pilot_phase_correction.py # Phase estimation
│   ├── constellation_demapper.py # Soft demodulation
│   ├── ldpc_decoder.py         # LDPC decoding
│   ├── bch_decoding.py         # BCH correction
│   ├── bb_deframer.py          # BB frame deframing
│   └── __init__.py
│
├── common/                      # Shared modules
│   ├── bit_interleaver.py      # DVB-S2 bit interleaver
│   ├── constellation_mapper.py # Constellation mapping
│   ├── pilot_insertion.py      # Pilot insertion
│   ├── pl_scrambler.py         # PL scrambling/descrambling
│   └── __init__.py
│
├── tests/                       # Test scripts and notebooks
│   ├── test_tx_rx_loopback.py  # TX→RX loopback test
│   ├── test_rx_loopback.py     # RX self-test
│   ├── demo_tx_rx.py           # Demo script
│   ├── crc8_decoder.py         # CRC8 utilities
│   ├── dvbs2x_full.py          # Full DVB-S2X demo
│   └── test.ipynb              # Jupyter notebook
│
├── data/                        # Input data
│   ├── GS_data/                # Ground Station data samples
│   │   └── umair_gs_bits.csv   # Test bits
│   ├── TS_data/                # Transport Stream data
│   └── Data/                   # Additional data files
│
├── config/                      # Configuration and matrices
│   ├── ldpc_matrices/          # LDPC parity matrices (*.mat)
│   │   └── dvbs2xLDPCParityMatrices.mat
│   ├── verification/           # Verification files
│   ├── archived_code/          # Old/unused implementations
│   └── verification.rar
│
├── results/                     # Output and results
│   ├── tx_run/                 # Transmitter output samples
│   │   ├── *_report.txt        # Reports
│   │   ├── *.txt               # Sample data (symbols, bits, etc.)
│   │   └── ...
│   ├── loopback/               # Loopback test results
│   │   ├── loopback_stats.json
│   │   ├── tx_bits_*.txt
│   │   ├── rx_bits_*.txt
│   │   └── ...
│   └── dvbs2x/                 # DVB-S2X output
│       └── *_report.txt
│
├── docs/                        # Documentation
│   ├── README.md               # Main documentation
│   ├── en_302307v010301a.pdf   # DVB-S2 standard
│   └── flow.txt                # Processing flow
│
├── __pycache__/                # Python cache (auto-generated)
│
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── .gitattributes             # Git attributes
```

## Quick Start

### 1. Install Dependencies
```bash
conda create -n dvbs2 python=3.10 numpy scipy matplotlib pandas
conda activate dvbs2
pip install bchlib scikit-dsp-comm pyldpc
```

### 2. Run Transmitter
```bash
cd tx/
python run_dvbs2.py
# Outputs: results/tx_run/
```

### 3. Run TX→RX Loopback Test
```bash
python tests/test_tx_rx_loopback.py --max-frames 3 --esn0-db 5
# Outputs: results/loopback/
```

### 4. View Results
```bash
cat results/loopback/loopback_stats.json
```

## Key Modules

### Transmitter (tx/)
- **run_dvbs2.py**: Interactive TX script - generates PLFRAME with pilots
- **BB_Frame.py**: BB frame construction, CRC-8
- **stream_adaptation.py**: Padding and BB scrambling  
- **bch_encoding.py**: BCH(Kbch, Nbch) encoding
- **ldpc_Encoding.py**: LDPC(Nbch, Ndim) encoding
- **pl_header.py**: PL header generation
- **bbframe_report.py**: Logging and reporting

### Receiver (rx/)
- **receiver_Chain.py**: Main RX orchestrator
- **pl_descrambler.py**: PL descrambling
- **pilot_phase_correction.py**: Pilot-aided phase estimation
- **constellation_demapper.py**: Soft LLR demodulation
- **ldpc_decoder.py**: LDPC normalized min-sum decoder
- **bch_decoding.py**: BCH error correction

### Common (common/)
- **bit_interleaver.py**: DVB-S2 bit interleaver
- **constellation_mapper.py**: QAM/APSK mapping
- **pl_scrambler.py**: PL scrambling
- **pilot_insertion.py**: Pilot symbol insertion

## Tests

### Available Tests
1. **test_tx_rx_loopback.py** - Full end-to-end test with AWGN
2. **test_rx_loopback.py** - RX chain validation
3. **dvbs2x_full.py** - Integrated demo
4. **test.ipynb** - Interactive notebook

### Running Tests
```bash
# Loopback with noise sweep
python tests/test_tx_rx_loopback.py --max-frames 5 --esn0-db 3

# Specific configuration
python tests/test_tx_rx_loopback.py \
  --fecframe normal \
  --rate 3/4 \
  --modulation 16APSK \
  --max-frames 10 \
  --esn0-db 8
```

## Output Formats

### Statistics JSON (loopback_stats.json)
```json
{
  "timestamp": "2026-02-04T...",
  "config": {...},
  "frames": [
    {
      "frame_num": 1,
      "tx_bits_shape": [720],
      "rx_bits_shape": [1000],
      "bits_compared": 720,
      "bit_errors": 0,
      "ber": 0.0,
      "frame_success": true
    }
  ],
  "summary": {
    "total_frames": 3,
    "successful_frames": 3,
    "overall_ber": 0.0
  }
}
```

## Git Workflow

```bash
# View recent commits
git log --oneline -5

# Stage and commit changes
git add .
git commit -m "Your message"

# Push to remote
git push origin main
```

## Notes
- LDPC matrices must be in `config/ldpc_matrices/`
- Input CSV must be in `data/GS_data/`
- All output saved automatically to `results/`
- Use `conda activate dvbs2` before running scripts
