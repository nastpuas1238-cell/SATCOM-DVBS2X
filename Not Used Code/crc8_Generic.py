import os
import sys
# Insert project root (one level up from this script) into sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from satcom.input_interface import read_and_process_input
# bitstream = read_and_process_input()

import numpy as np

# DVB-S2 CRC-8 generator polynomial:
# g(X) = X^8 + X^7 + X^6 + X^4 + X^2 + 1  →  0xD5 (binary 1101 0101)
CRC8_POLY = 0xD5  


def dvbs2_crc8(bitstream: np.ndarray) -> int:
    """
    Compute DVB-S2 CRC-8 on the input bitstream u(X)
    (bitstream must EXCLUDE the sync byte).
    
    Input: numpy array of 0/1 bits (length = UPL - 8)

    Output: Integer CRC-8 value (0–255)
    """
    crc = 0  # shift register initialized to all zeros

    for bit in bitstream:
        msb = (crc >> 7) & 1        # Extract MSB of CRC
        xor_input = msb ^ bit       # XOR incoming bit with MSB
        crc = ((crc << 1) & 0xFF)   # Shift left
        
        if xor_input == 1:
            crc ^= CRC8_POLY        # XOR polynomial when xor_input = 1

    return crc

# Example dummy payload (bitstream EXCLUDING sync byte)
data = np.array([1,0,1,1,0,1,0,0, 1,1,0,0,1,0,1,1], dtype=np.uint8)

crc = dvbs2_crc8(data)
print("Computed DVB-S2 CRC-8 =", hex(crc))




