#!/usr/bin/env python3
"""
DVBS‑2 Transmitter Script

Refactored for:
- Better structure and readability
- Safer file and path handling
- Easier configuration and testing
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ------------------------------------------------------------
# Import modules
# ------------------------------------------------------------

from BB_Frame import (
    dvbs2_bbframe_generator_from_bits_csv,
    build_bbheader,
    PacketizedCrc8Stream,
    compute_syncd_packetized,
    load_bits_csv,
    resolve_input_path,
)

from stream_adaptation import (
    get_kbch,
    pad_bbframe_rate,
    save_bbframe_to_file_rate,
    stream_adaptation_rate,
)

from bbframe_report import BBFrameReport
from bch_encoding import BCH_PARAMS, bch_encode_bbframe
from ldpc_Encoding import DVB_LDPC_Encoder

from bit_interleaver import dvbs2_bit_interleave, dvbs2_bit_deinterleave
from constellation_mapper import dvbs2_constellation_map


# ------------------------------------------------------------
# Default user configuration (can be changed in one place)
# ------------------------------------------------------------

BITS_CSV_PATH = Path(
    r"C:\Users\umair\Videos\JOB - NASTP\SATCOM\Code\GS_data\umair_gs_bits.csv"
)
MAX_FRAMES = 3
REPORT_FILE = Path("dvbs2_full_report.txt")
MAT_PATH = Path(
    r"C:\Users\umair\Videos\JOB - NASTP\SATCOM\Code\s2xLDPCParityMatrices\dvbs2xLDPCParityMatrices.mat"
)


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def write_bits_single_line(path: Path, bits: np.ndarray) -> None:
    """
    Write bits as a single line of '0'/'1' characters.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("".join("1" if b else "0" for b in bits))


def write_symbols(path: Path, syms: np.ndarray) -> None:
    """
    Write complex symbols as 'real imag' per line.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in syms:
            f.write(f"{s.real:+.6f} {s.imag:+.6f}\n")


def get_user_inputs() -> tuple[str, str, str, str, float, int, int, int]:
    """
    Collect and validate user inputs from stdin.

    Returns:
        stream_type, fecframe, rate, modulation, alpha, DFL, UPL, SYNC
    """
    stream_type = input("Enter stream type (TS or GS): ").strip().upper()
    if stream_type not in {"TS", "GS"}:
        raise ValueError("stream_type must be 'TS' or 'GS'")

    fecframe = input("Enter FECFRAME type (normal/short): ").strip().lower()
    if fecframe not in {"normal", "short"}:
        raise ValueError("FECFRAME must be 'normal' or 'short'")

    rate = input(
        "Enter code rate (e.g., 1/2, 3/5, 2/3, 3/4, 5/6, 8/9, 9/10): "
    ).strip()

    modulation = input(
        "Enter modulation (QPSK, 8PSK, 16APSK, 32APSK): "
    ).strip().upper()
    if modulation not in {"QPSK", "8PSK", "16APSK", "32APSK"}:
        raise ValueError("Invalid modulation")

    alpha = float(input("Enter roll-off alpha (0.35 / 0.25 / 0.20): ").strip())
    if alpha not in {0.35, 0.25, 0.20}:
        raise ValueError("alpha must be 0.35, 0.25, or 0.20")

    DFL = int(input("Enter DFL: ").strip())
    if DFL < 0:
        raise ValueError("DFL must be non-negative")

    if stream_type == "TS":
        UPL = 188 * 8
        SYNC = 0x47
    else:
        UPL = int(input("Enter UPL in bits (0 for continuous GS): ").strip())
        SYNC = int(input("Enter SYNC byte in hex (e.g., 47): ").strip(), 16)

    return stream_type, fecframe, rate, modulation, alpha, DFL, UPL, SYNC


# ------------------------------------------------------------
# MAIN RUN
# ------------------------------------------------------------

def run_dvbs2_transmitter(
    bits_csv_path: Path = BITS_CSV_PATH,
    max_frames: int = MAX_FRAMES,
    report_file: Path = REPORT_FILE,
    mat_path: Path = MAT_PATH,
) -> None:
    """
    Execute the DVBS-2 transmitter chain.

    All external paths are passed in as parameters to make the
    function easier to test and reuse.
    """
    print("\n========== DVB-S2 TRANSMITTER RUN ==========" "\n")

    # Ensure paths are Path objects
    bits_csv_path = Path(bits_csv_path)
    report_file = Path(report_file)
    mat_path = Path(mat_path)

    # Basic file existence checks
    if not bits_csv_path.exists():
        raise FileNotFoundError(f"Bits CSV not found: {bits_csv_path}")
    if not mat_path.exists():
        raise FileNotFoundError(f"LDPC parity matrix file not found: {mat_path}")

    # Report object
    report = BBFrameReport(str(report_file))

    # -----------------------------
    # User inputs
    # -----------------------------
    (
        stream_type,
        fecframe,
        rate,
        modulation,
        alpha,
        DFL,
        UPL,
        SYNC,
    ) = get_user_inputs()

    # Validate DFL against Kbch
    Kbch = get_kbch(fecframe, rate)
    if not (0 <= DFL <= Kbch - 80):
        raise ValueError(f"DFL must satisfy 0 ≤ DFL ≤ {Kbch - 80}")

    # -----------------------------
    # Load input bits
    # -----------------------------
    csv_path = resolve_input_path(str(bits_csv_path))
    in_bits = load_bits_csv(csv_path)
    report.log_input_data(csv_path, in_bits)

    # -----------------------------
    # Mode adaptation setup
    # -----------------------------
    if UPL == 0:
        # Continuous GS
        mode_stream = in_bits
        stream_ptr = 0
        packetized = False
        df_global_pos: Optional[int] = None
    else:
        # Packetized (TS or GS with UPL > 0)
        crc_stream = PacketizedCrc8Stream(in_bits, UPL)
        packetized = True
        df_global_pos = 0

    frames = 0

    # -----------------------------
    # FRAME LOOP
    # -----------------------------
    while frames < max_frames:

        # ----- DATA FIELD -----
        if DFL == 0:
            DF = np.array([], dtype=np.uint8)
        elif packetized:
            DF = crc_stream.read_bits(DFL)
        else:
            DF = mode_stream[stream_ptr : stream_ptr + DFL]
            if DF.size < DFL:
                # Not enough data left; stop gracefully
                print("Not enough bits left to fill DF; stopping.")
                break
            stream_ptr += DFL

        if packetized:
            SYNCD = compute_syncd_packetized(df_global_pos, DFL, UPL)
            df_global_pos += DFL
        else:
            SYNCD = 0

        BBHEADER = build_bbheader(
            matype1=0x00,
            matype2=0x00,
            upl=UPL,
            dfl=DFL,
            sync=SYNC,
            syncd=SYNCD,
        )

        BBFRAME = np.concatenate([BBHEADER, DF])
        frames += 1

        # Log BBFRAME / header
        report.log_bbheader(BBHEADER, 0x00, UPL, DFL, SYNC, SYNCD)
        report.log_merger_slicer(DF, DFL)
        report.log_bbframe(BBFRAME)

        # -----------------------------
        # STREAM ADAPTATION
        # -----------------------------
        padded_bbframe = pad_bbframe_rate(BBFRAME, fecframe, rate)
        adapted = stream_adaptation_rate(BBFRAME, fecframe, rate)

        # -----------------------------
        # BCH ENCODING
        # -----------------------------
        Kbch_val, Nbch, t = BCH_PARAMS[(fecframe, rate)]
        bch_codeword = bch_encode_bbframe(adapted, fecframe, rate)
        report.log_bch_encoding(
            adapted, bch_codeword, fecframe, rate, Kbch_val, Nbch, t
        )

        # -----------------------------
        # LDPC ENCODING
        # -----------------------------
        ldpc_encoder = DVB_LDPC_Encoder(str(mat_path))
        ldpc_codeword = ldpc_encoder.encode(bch_codeword, fecframe, rate)

        report.section("LDPC ENCODING")
        report.bits("LDPC codeword", ldpc_codeword)

        # -----------------------------
        # BIT INTERLEAVING (ETSI 5.3.3)
        # -----------------------------
        interleaved = dvbs2_bit_interleave(ldpc_codeword, modulation)

        report.section("BIT INTERLEAVING (ETSI 5.3.3)")
        report.bits("Interleaved bits", interleaved)

        # -----------------------------
        # CONSTELLATION MAPPING (ETSI 5.4)
        # -----------------------------
        symbols = dvbs2_constellation_map(
            interleaved,
            modulation,
            code_rate=rate,
        )

        report.section("CONSTELLATION MAPPING (ETSI 5.4)")
        report.write(f"Modulation     : {modulation}")
        report.write(f"Total symbols  : {len(symbols)}")
        report.write(
            "First symbols  : "
            + ", ".join(f"{s.real:+.3f}{s.imag:+.3f}j" for s in symbols[:8])
        )

        # -----------------------------
        # SANITY CHECK
        # -----------------------------
        recovered = dvbs2_bit_deinterleave(interleaved, modulation)
        if not np.array_equal(recovered, ldpc_codeword):
            report.write("WARNING: Interleaver round-trip mismatch")
        else:
            report.write("Interleaver round-trip: OK")

        # -----------------------------
        # SAVE OUTPUT FILES
        # -----------------------------
        out_base = report_file.parent  # keep all outputs next to report

        write_bits_single_line(out_base / f"ldpc_{frames}.txt", ldpc_codeword)
        write_bits_single_line(
            out_base / f"interleaved_{frames}.txt", interleaved
        )
        write_symbols(out_base / f"symbols_{frames}.txt", symbols)

        save_bbframe_to_file_rate(
            padded_bbframe,
            adapted,
            str(out_base / f"bbframe_{frames}.txt"),
            fecframe,
            rate,
            output_mode="both_single_line",
        )

        print(f"Frame {frames}: OK")

    report.close()
    print(f"\nReport written to: {report_file}")


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

if __name__ == "__main__":
    try:
        run_dvbs2_transmitter()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

