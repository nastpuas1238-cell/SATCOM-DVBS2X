"""
End-to-end DVB-S2X-like (S2 baseline) TX→RX using the same input data path as tx/run_dvbs2.py.
Non-interactive, single-frame demo with real bits from GS_data/umair_gs_bits.csv.
Config: QPSK, rate 1/2, short frame, pilots on, scrambling_code=0, DFL capped to fit Kbch.
Outputs saved to dvbs2x_output/.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import json

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# TX imports
from tx.BB_Frame import build_bbheader, load_bits_csv, resolve_input_path
from tx.stream_adaptation import stream_adaptation_rate, get_kbch
from tx.bch_encoding import bch_encode_bbframe
from tx.ldpc_Encoding import DVB_LDPC_Encoder
from common.bit_interleaver import dvbs2_bit_interleave
from common.constellation_mapper import dvbs2_constellation_map
from common.pilot_insertion import insert_pilots_into_payload
from tx.pl_header import build_plheader, modcod_from_modulation_rate
from common.pl_scrambler import pl_scramble_full_plframe

# RX imports
from rx.receiver_Chain import process_rx_plframe


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_bits(path: str, bits: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join("1" if int(b) else "0" for b in bits.reshape(-1)))


def save_symbols(path: str, syms: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in syms.reshape(-1):
            f.write(f"{s.real:+.6f} {s.imag:+.6f}\n")

def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def run_end_to_end() -> None:
    # Configuration aligned with tx/run_dvbs2.py defaults
    BITS_CSV_PATH = r"C:\Users\umair\Videos\JOB - NASTP\SATCOM\Code\GS_data\umair_gs_bits.csv"
    MAT_PATH = r"C:\Users\umair\Videos\JOB - NASTP\SATCOM\Code\s2xLDPCParityMatrices\dvbs2xLDPCParityMatrices.mat"
    fecframe = "short"
    rate = "1/2"
    modulation = "QPSK"
    pilots_on = True
    scrambling_code = 0
    dfl = 6000  # data field length bits (<= Kbch-80)

    out_dir = ensure_dir(os.path.join(ROOT, "dvbs2x_output"))

    # Load input bits
    bits_path = resolve_input_path(BITS_CSV_PATH)
    in_bits = load_bits_csv(bits_path)

    Kbch = get_kbch(fecframe, rate)
    dfl = min(dfl, Kbch - 80, in_bits.size)  # safety cap

    df_bits = in_bits[:dfl]

    # BBHEADER (GS continuous: UPL=0)
    BBHEADER = build_bbheader(
        matype1=0x00,
        matype2=0x00,
        upl=0,
        dfl=dfl,
        sync=0x00,
        syncd=0x0000,
    )
    BBFRAME = np.concatenate([BBHEADER, df_bits])

    # Stream adaptation
    scrambled = stream_adaptation_rate(BBFRAME, fecframe, rate)

    # BCH
    bch_codeword = bch_encode_bbframe(scrambled, fecframe, rate)

    # LDPC
    ldpc_encoder = DVB_LDPC_Encoder(MAT_PATH)
    ldpc_codeword = ldpc_encoder.encode(bch_codeword, fecframe, rate)

    # Interleave + map
    interleaved = dvbs2_bit_interleave(ldpc_codeword, modulation)
    payload_syms = dvbs2_constellation_map(interleaved, modulation, code_rate=rate)

    # PLHEADER
    modcod = modcod_from_modulation_rate(modulation, rate)
    _, plh_syms = build_plheader(modcod, fecframe, pilots=pilots_on)

    # Pilots
    payload_with_pilots, _ = insert_pilots_into_payload(payload_syms, pilots_on, fecframe=fecframe)

    # PLFRAME pre-scramble
    plframe_pre = np.concatenate([plh_syms, payload_with_pilots])
    plframe = pl_scramble_full_plframe(plframe_pre, scrambling_code=scrambling_code, plheader_len=len(plh_syms))

    # Save TX artifacts
    save_bits(os.path.join(out_dir, "df_bits.txt"), df_bits)
    save_bits(os.path.join(out_dir, "BBFRAME.txt"), BBFRAME)
    save_bits(os.path.join(out_dir, "scrambled.txt"), scrambled)
    save_bits(os.path.join(out_dir, "bch_codeword.txt"), bch_codeword)
    save_bits(os.path.join(out_dir, "ldpc_codeword.txt"), ldpc_codeword)
    save_bits(os.path.join(out_dir, "interleaved_bits.txt"), interleaved)
    save_symbols(os.path.join(out_dir, "payload_symbols.txt"), payload_syms)
    save_symbols(os.path.join(out_dir, "plh_symbols.txt"), plh_syms)
    save_symbols(os.path.join(out_dir, "plframe_symbols.txt"), plframe)
    tx_meta_paths = {
        "df_bits": "df_bits.txt",
        "BBFRAME": "BBFRAME.txt",
        "scrambled": "scrambled.txt",
        "bch_codeword": "bch_codeword.txt",
        "ldpc_codeword": "ldpc_codeword.txt",
        "interleaved_bits": "interleaved_bits.txt",
        "payload_symbols": "payload_symbols.txt",
        "plh_symbols": "plh_symbols.txt",
        "plframe_symbols": "plframe_symbols.txt",
    }

    # RX
    rx_out = process_rx_plframe(
        plframe,
        fecframe=fecframe,
        scrambling_code=scrambling_code,
        modulation=modulation,
        rate=rate,
        noise_var=1e-6,
        decode_ldpc=True,
        ldpc_max_iter=30,
    )

    df_rx = rx_out["df_bits"]
    if df_rx is None or df_rx.size != df_bits.size:
        raise AssertionError("Receiver failed to recover DF bits")
    if not np.array_equal(df_bits, df_rx):
        errs = int(np.sum(df_bits != df_rx))
        raise AssertionError(f"DF mismatch: {errs} errors")

    # Save RX artifacts
    save_symbols(os.path.join(out_dir, "payload_corrected.txt"), rx_out["payload_corrected"])
    save_symbols(os.path.join(out_dir, "pilots_extracted.txt"), rx_out["pilots"].reshape(-1))
    np.savetxt(os.path.join(out_dir, "phase_estimates.txt"), rx_out["phase_estimates"])
    np.savetxt(os.path.join(out_dir, "llrs_interleaved.txt"), rx_out["llrs_interleaved"])
    np.savetxt(os.path.join(out_dir, "llrs_deinterleaved.txt"), rx_out["llrs_deinterleaved"])
    if rx_out["ldpc_bits"] is not None:
        save_bits(os.path.join(out_dir, "ldpc_bits_rx.txt"), rx_out["ldpc_bits"])
    if rx_out["bch_payload"] is not None:
        save_bits(os.path.join(out_dir, "bch_payload_rx.txt"), rx_out["bch_payload"])
    if rx_out["bbframe_padded"] is not None:
        save_bits(os.path.join(out_dir, "bbframe_padded_rx.txt"), rx_out["bbframe_padded"])
    if rx_out["df_bits"] is not None:
        save_bits(os.path.join(out_dir, "df_bits_rx.txt"), rx_out["df_bits"])
    # Save additional raw payload and metadata
    save_symbols(os.path.join(out_dir, "payload_raw.txt"), rx_out["payload_raw"])
    save_json(os.path.join(out_dir, "pilot_meta.json"), rx_out["pilot_meta"])
    save_json(os.path.join(out_dir, "phase_meta.json"), rx_out["phase_meta"])
    save_json(os.path.join(out_dir, "demap_meta.json"), rx_out["demap_meta"])
    save_json(os.path.join(out_dir, "ldpc_meta.json"), rx_out["ldpc_meta"])
    if rx_out["df_meta"] is not None:
        save_json(os.path.join(out_dir, "df_meta.json"), rx_out["df_meta"])

    rx_meta_paths = {
        "payload_corrected": "payload_corrected.txt",
        "payload_raw": "payload_raw.txt",
        "pilots_extracted": "pilots_extracted.txt",
        "phase_estimates": "phase_estimates.txt",
        "llrs_interleaved": "llrs_interleaved.txt",
        "llrs_deinterleaved": "llrs_deinterleaved.txt",
        "ldpc_bits_rx": "ldpc_bits_rx.txt" if rx_out["ldpc_bits"] is not None else None,
        "bch_payload_rx": "bch_payload_rx.txt" if rx_out["bch_payload"] is not None else None,
        "bbframe_padded_rx": "bbframe_padded_rx.txt" if rx_out["bbframe_padded"] is not None else None,
        "df_bits_rx": "df_bits_rx.txt" if rx_out["df_bits"] is not None else None,
        "pilot_meta": "pilot_meta.json",
        "phase_meta": "phase_meta.json",
        "demap_meta": "demap_meta.json",
        "ldpc_meta": "ldpc_meta.json",
        "df_meta": "df_meta.json" if rx_out["df_meta"] is not None else None,
    }

    # Report
    report_path = os.path.join(out_dir, "dvbs2x_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DVB-S2X TX→RX Report\n")
        f.write("====================\n")
        f.write(f"Input bits file      : {bits_path}\n")
        f.write(f"Modulation           : {modulation}\n")
        f.write(f"Rate                 : {rate}\n")
        f.write(f"FECFRAME             : {fecframe}\n")
        f.write(f"Pilots               : {pilots_on}\n")
        f.write(f"Scrambling code      : {scrambling_code}\n")
        f.write(f"DFL                  : {dfl}\n")
        f.write(f"LDPC MAT             : {MAT_PATH}\n")
        f.write("\nLengths:\n")
        f.write(f"DF bits              : {df_bits.size}\n")
        f.write(f"BBFRAME              : {BBFRAME.size}\n")
        f.write(f"BCH codeword         : {bch_codeword.size}\n")
        f.write(f"LDPC codeword        : {ldpc_codeword.size}\n")
        f.write(f"Interleaved bits     : {interleaved.size}\n")
        f.write(f"Payload symbols      : {payload_syms.size}\n")
        f.write(f"PLFRAME symbols      : {plframe.size}\n")
        f.write("\nRX stats:\n")
        f.write(f"LDPC iterations      : {rx_out['ldpc_meta']['iterations']}\n")
        f.write(f"LDPC syndrome wgt    : {rx_out['ldpc_meta']['syndrome_weight']}\n")
        f.write(f"BCH corrected?       : {rx_out['bch_meta']['corrected'] if rx_out['bch_meta'] else 'n/a'}\n")
        f.write(f"Recovered DF bits    : {df_rx.size}\n")
        f.write("\nResult: PASS (DF match)\n")
        f.write("\nTX files:\n")
        for k, v in tx_meta_paths.items():
            f.write(f"  {k:18s}: {v}\n")
        f.write("\nRX files:\n")
        for k, v in rx_meta_paths.items():
            if v:
                f.write(f"  {k:18s}: {v}\n")

    print("DVB-S2X TX->RX completed successfully.")
    print(f"Artifacts in: {out_dir}")


if __name__ == "__main__":
    run_end_to_end()
