# crc8_decoder.py
# =============================================================================
# DVB-S2 CRC-8 Decoder
# ETSI EN 302 307 V1.3.1
# Clause 5.1.6: Mode Adaptation
#
# Polynomial: x^8 + x^2 + x + 1  (0x07)
# Init value: 0x00
# No reflection, no final XOR
# =============================================================================

from __future__ import annotations
import numpy as np
from typing import Tuple


# -----------------------------------------------------------------------------
# CRC-8 parameters (ETSI DVB-S2)
# -----------------------------------------------------------------------------

CRC8_POLY = 0x07
CRC8_INIT = 0x00
CRC8_WIDTH = 8


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _bits_to_bytes(bits: np.ndarray) -> np.ndarray:
    """
    Convert bit array (0/1) to byte array (MSB-first).
    """
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size % 8 != 0:
        raise ValueError("Bit length must be multiple of 8")

    bytes_out = np.zeros(bits.size // 8, dtype=np.uint8)
    for i in range(bytes_out.size):
        b = 0
        for k in range(8):
            b = (b << 1) | bits[i * 8 + k]
        bytes_out[i] = b
    return bytes_out


def _crc8_compute(data_bytes: np.ndarray) -> int:
    """
    Compute CRC-8 over byte array using DVB-S2 polynomial.
    """
    crc = CRC8_INIT
    for byte in data_bytes:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


# -----------------------------------------------------------------------------
# Core: CRC-8 decoder
# -----------------------------------------------------------------------------

def crc8_decode_packet(
    packet_bits: np.ndarray
) -> Tuple[np.ndarray, bool, int, int]:
    """
    Decode and check CRC-8 from a packetized stream unit.

    Parameters
    ----------
    packet_bits : np.ndarray (uint8)
        Bits including payload + CRC byte (last 8 bits).

    Returns
    -------
    payload_bits : np.ndarray
        Payload bits with CRC removed.
    crc_ok : bool
        True if CRC matches.
    crc_rx : int
        Received CRC value.
    crc_calc : int
        Recomputed CRC value.
    """
    packet_bits = np.asarray(packet_bits, dtype=np.uint8).reshape(-1)
    if packet_bits.size < 8 or packet_bits.size % 8 != 0:
        raise ValueError("Packet length must be >= 8 bits and byte-aligned")

    payload_bits = packet_bits[:-8]
    crc_bits = packet_bits[-8:]

    payload_bytes = _bits_to_bytes(payload_bits)
    crc_rx = int(_bits_to_bytes(crc_bits)[0])

    crc_calc = _crc8_compute(payload_bytes)

    crc_ok = (crc_rx == crc_calc)

    return payload_bits, crc_ok, crc_rx, crc_calc


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test():
    # Example payload
    payload_bits = np.array(
        [1,0,1,1,0,0,1,0,
         0,1,0,1,1,0,0,1],
        dtype=np.uint8
    )

    payload_bytes = _bits_to_bytes(payload_bits)
    crc = _crc8_compute(payload_bytes)

    crc_bits = np.array(
        [(crc >> (7 - i)) & 1 for i in range(8)],
        dtype=np.uint8
    )

    packet = np.concatenate([payload_bits, crc_bits])

    decoded_payload, ok, rx, calc = crc8_decode_packet(packet)

    assert ok
    assert rx == calc
    assert np.array_equal(decoded_payload, payload_bits)

    print("CRC-8 decoder self-test PASSED")


if __name__ == "__main__":
    _self_test()
