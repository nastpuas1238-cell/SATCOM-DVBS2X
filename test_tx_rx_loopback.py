"""
TX→RX Loopback Test

This script:
1. Runs the transmitter (tx/run_dvbs2.py) to generate a full PLFRAME with pilots
2. Optionally adds AWGN noise
3. Feeds the output directly into the receiver chain (rx/receiver_Chain.py)
4. Compares TX input bits with RX decoded output bits
5. Generates performance metrics (BER, packet success rate)
"""

import os
import sys
import numpy as np
import json
from datetime import datetime

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


def write_bits_single_line(path: str, bits: np.ndarray):
    """Write bits (0/1) as single line separated by spaces."""
    bits_str = " ".join(str(int(b)) for b in bits.flatten())
    with open(path, "w") as f:
        f.write(bits_str + "\n")


def run_tx_rx_loopback(
    fecframe: str = "short",
    rate: str = "1/2",
    modulation: str = "QPSK",
    pilots_on: bool = True,
    scrambling_code: int = 0,
    esn0_db: float | None = None,
    max_frames: int = 1,
    output_dir: str = "loopback_output",
) -> dict:
    """
    Run a complete TX→RX loopback test.
    
    Args:
        fecframe: "normal" or "short"
        rate: Code rate (e.g., "1/2", "3/5", "2/3", "3/4", "5/6", "8/9", "9/10")
        modulation: "QPSK", "8PSK", "16APSK", "32APSK"
        pilots_on: Enable pilot insertion
        scrambling_code: PL scrambling code (0..262142)
        esn0_db: SNR in dB (None for noiseless)
        max_frames: Number of frames to process
        output_dir: Output directory for results
        
    Returns:
        Dictionary with statistics
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get parameters
    Kbch = get_kbch(fecframe, rate)
    DFL = min(1000, Kbch - 80)  # Use reasonable data field length
    UPL = 120
    SYNC_BYTE = 0x47
    
    # Get CSV path
    csv_path = resolve_input_path(os.path.join(ROOT, "GS_data", "umair_gs_bits.csv"))
    in_bits = load_bits_csv(csv_path)
    
    # Get LDPC encoder
    mat_path = os.path.join(ROOT, "s2xLDPCParityMatrices", "dvbs2xLDPCParityMatrices.mat")
    ldpc_encoder = DVB_LDPC_Encoder(mat_path)
    
    # Statistics
    stats = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "fecframe": fecframe,
            "rate": rate,
            "modulation": modulation,
            "pilots_on": pilots_on,
            "scrambling_code": scrambling_code,
            "esn0_db": esn0_db,
            "max_frames": max_frames,
        },
        "frames": []
    }
    
    # Process each frame
    for frame_num in range(1, max_frames + 1):
        print(f"\n--- Frame {frame_num} ---")
        
        # =====================================================================
        # TX SIDE: Generate PLFRAME
        # =====================================================================
        
        # Get TX input bits (for later comparison)
        tx_input_bits = in_bits[:DFL] if DFL > 0 else np.array([], dtype=np.uint8)
        
        # Build BB header
        bbheader_bits = build_bbheader(
            matype1=0x00,
            matype2=0x00,
            upl=UPL,
            dfl=DFL,
            sync=SYNC_BYTE,
            syncd=0
        )
        
        # Build BB frame (with padding)
        if len(tx_input_bits) > 0:
            bbframe = np.concatenate([bbheader_bits, tx_input_bits])
        else:
            bbframe = bbheader_bits
        
        adapted = stream_adaptation_rate(bbframe, fecframe, rate)
        
        # BCH encode
        bch_codeword = bch_encode_bbframe(adapted, fecframe, rate=rate)
        
        # LDPC encode
        ldpc_codeword = ldpc_encoder.encode(bch_codeword, fecframe, rate)
        
        # Bit interleave
        interleaved = dvbs2_bit_interleave(ldpc_codeword, modulation)
        
        # Constellation map
        symbols = dvbs2_constellation_map(interleaved, modulation)
        
        # Get MODCOD for PL header
        modcod = modcod_from_modulation_rate(modulation, rate)
        
        # Build PL header
        _, plh_syms = build_plheader(modcod, fecframe, pilots=pilots_on)
        
        # Insert pilots
        payload_with_pilots, _ = insert_pilots_into_payload(symbols, pilots_on=pilots_on, fecframe=fecframe)
        
        # Concatenate PL header + payload with pilots
        plframe_tx = np.concatenate([plh_syms, payload_with_pilots])
        
        # PL scramble
        plframe_scrambled = pl_scramble_full_plframe(plframe_tx, scrambling_code=scrambling_code, plheader_len=len(plh_syms))
        
        print(f"TX: Generated PLFRAME shape={plframe_scrambled.shape}, pilots={pilots_on}")
        
        # =====================================================================
        # CHANNEL SIMULATION: Add AWGN noise if specified
        # =====================================================================
        
        if esn0_db is not None:
            # Power of QPSK symbols is 1
            signal_power = 1.0
            noise_power = signal_power / (10 ** (esn0_db / 10))
            noise = np.sqrt(noise_power / 2) * (np.random.randn(len(plframe_scrambled)) + 1j * np.random.randn(len(plframe_scrambled)))
            plframe_rx = plframe_scrambled + noise
            print(f"Channel: Added AWGN, Es/N0={esn0_db} dB, noise_power={noise_power:.6f}")
        else:
            plframe_rx = plframe_scrambled
            noise_power = 1e-10  # Very small noise for numerical stability
            print("Channel: Noiseless (numerical noise_var=1e-10)")
        
        # =====================================================================
        # RX SIDE: Process received PLFRAME
        # =====================================================================
        
        rx_output = process_rx_plframe(
            plframe_rx,
            fecframe=fecframe,
            scrambling_code=scrambling_code,
            modulation=modulation,
            rate=rate,
            noise_var=noise_power,
            ldpc_mat_path=mat_path,
            ldpc_max_iter=30,
            ldpc_norm=0.75,
            decode_ldpc=True,
        )
        
        # =====================================================================
        # EVALUATE RESULTS
        # =====================================================================
        
        rx_df_bits = rx_output.get("df_bits")
        
        frame_stats = {
            "frame_num": frame_num,
            "tx_bits_shape": tx_input_bits.shape,
        }
        
        if rx_df_bits is not None:
            # Compare TX input with RX output
            min_len = min(len(tx_input_bits), len(rx_df_bits))
            tx_cropped = tx_input_bits[:min_len]
            rx_cropped = rx_df_bits[:min_len]
            
            errors = np.sum(tx_cropped != rx_cropped)
            ber = errors / min_len if min_len > 0 else 1.0
            success = errors == 0
            
            frame_stats.update({
                "rx_bits_shape": rx_df_bits.shape,
                "bits_compared": min_len,
                "bit_errors": int(errors),
                "ber": float(ber),
                "frame_success": bool(success),
            })
            
            print(f"RX: Decoded {len(rx_df_bits)} bits")
            print(f"    Bit Errors: {errors}/{min_len}")
            print(f"    BER: {ber:.6e}")
            print(f"    Frame Success: {success}")
        else:
            frame_stats["error"] = "RX decoding failed"
            print("RX: Decoding FAILED")
        
        stats["frames"].append(frame_stats)
        
        # Save artifacts
        write_bits_single_line(
            os.path.join(output_dir, f"tx_bits_frame{frame_num}.txt"),
            tx_input_bits
        )
        if rx_df_bits is not None:
            write_bits_single_line(
                os.path.join(output_dir, f"rx_bits_frame{frame_num}.txt"),
                rx_df_bits
            )
    
    # =====================================================================
    # SUMMARY STATISTICS
    # =====================================================================
    
    total_frames = len(stats["frames"])
    successful_frames = sum(1 for f in stats["frames"] if f.get("frame_success", False))
    total_errors = sum(f.get("bit_errors", 0) for f in stats["frames"])
    total_bits = sum(f.get("bits_compared", 0) for f in stats["frames"])
    
    stats["summary"] = {
        "total_frames": total_frames,
        "successful_frames": successful_frames,
        "frame_success_rate": successful_frames / total_frames if total_frames > 0 else 0.0,
        "total_bit_errors": int(total_errors),
        "total_bits_tested": int(total_bits),
        "overall_ber": total_errors / total_bits if total_bits > 0 else 1.0,
    }
    
    # Save statistics
    stats_file = os.path.join(output_dir, "loopback_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*60)
    print("LOOPBACK TEST SUMMARY")
    print("="*60)
    print(f"Frames Processed: {total_frames}")
    print(f"Successful Frames: {successful_frames}/{total_frames}")
    print(f"Frame Success Rate: {stats['summary']['frame_success_rate']:.2%}")
    print(f"Total Bit Errors: {total_errors}/{total_bits}")
    print(f"Overall BER: {stats['summary']['overall_ber']:.6e}")
    print(f"Results saved to: {output_dir}/")
    print("="*60)
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TX→RX Loopback Test")
    parser.add_argument("--fecframe", default="short", help="normal or short")
    parser.add_argument("--rate", default="1/2", help="Code rate")
    parser.add_argument("--modulation", default="QPSK", help="Modulation scheme")
    parser.add_argument("--no-pilots", action="store_true", help="Disable pilots")
    parser.add_argument("--scrambling-code", type=int, default=0, help="PL scrambling code")
    parser.add_argument("--esn0-db", type=float, default=None, help="Es/N0 in dB (None for noiseless)")
    parser.add_argument("--max-frames", type=int, default=3, help="Number of frames")
    parser.add_argument("--output-dir", default="loopback_output", help="Output directory")
    
    args = parser.parse_args()
    
    stats = run_tx_rx_loopback(
        fecframe=args.fecframe,
        rate=args.rate,
        modulation=args.modulation,
        pilots_on=not args.no_pilots,
        scrambling_code=args.scrambling_code,
        esn0_db=args.esn0_db,
        max_frames=args.max_frames,
        output_dir=args.output_dir,
    )
