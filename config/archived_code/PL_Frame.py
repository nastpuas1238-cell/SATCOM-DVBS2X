import numpy as np
import pandas as pd


# ============================================================
#  DVB-S2 CONSTANTS
# ============================================================

CRC8_POLY = 0xD5


# Common Kbch values (normal/short FECFRAME)

KBCH_NORMAL = 64800
KBCH_SHORT  = 16200



# ============================================================
#  Utility: Convert string → numpy array of 0/1 bits
# ============================================================

def convert_to_bits(data: str) -> np.ndarray:

    bits = ''.join(format(ord(c), '08b') for c in data)

    return np.array([int(b) for b in bits], dtype=np.uint8)



# ============================================================
#  CRC-8 (DVB-S2, §5.1.4)
# ============================================================

def dvbs2_crc8(bitstream: np.ndarray) -> int:

    crc = 0

    for bit in bitstream:

        msb = (crc >> 7) & 1

        xor_in = msb ^ bit

        crc = ((crc << 1) & 0xFF)

        if xor_in:
            crc ^= CRC8_POLY

    return crc



# ============================================================
#  TS PACKET GENERATION (188 bytes)
# ============================================================

def generate_ts_packet(data_bits: np.ndarray, sync_byte: int) -> np.ndarray:

    payload_bits_needed = 187 * 8

    if len(data_bits) < payload_bits_needed:

        payload = np.pad(data_bits, (0, payload_bits_needed - len(data_bits)))

    else:

        payload = data_bits[:payload_bits_needed]


    sync_bits = np.array([int(b) for b in format(sync_byte, '08b')], dtype=np.uint8)

    return np.concatenate([sync_bits, payload])



# ============================================================
#  GS PACKET GENERATION
# ============================================================

def generate_gs_packet(data_bits: np.ndarray, upl: int, sync_byte: int):

    if upl == 0:

        # continuous stream, no sync/UPL structure
        return data_bits


    payload_bits_needed = upl - 8   # 8 bits reserved for sync


    if len(data_bits) < payload_bits_needed:

        payload = np.pad(data_bits, (0, payload_bits_needed - len(data_bits)))

    else:

        payload = data_bits[:payload_bits_needed]


    sync_bits = np.array([int(b) for b in format(sync_byte, '08b')], dtype=np.uint8)

    return np.concatenate([sync_bits, payload])



# ============================================================
#  FIND SYNCd (TS MODE)
# ============================================================

def compute_sync_distance(data_field: np.ndarray):
    """
    Correct SYNCd detection:
    - A TS UP always begins at a multiple of 8 bits.
    - We must check only byte-aligned positions.
    - This avoids false sync matches inside random bit patterns.
    """

    sync_bits = np.array([0,1,0,0,0,1,1,1], dtype=np.uint8)   # 0x47

    # Check only byte-aligned positions → i = 0, 8, 16, 24, ...
    for i in range(0, len(data_field) - 7, 8):
        if np.array_equal(data_field[i:i+8], sync_bits):
            return i

    # If no aligned sync found → SYNCd = 0
    return 0




# ============================================================
#  SLICER — Extract exactly DFL bits (continuous or packetized)
# ============================================================

def slice_stream(bitstream: np.ndarray, DFL: int):

    if len(bitstream) < DFL:

        padded = np.pad(bitstream, (0, DFL - len(bitstream)))

        return padded, np.array([], dtype=np.uint8)


    else:

        return bitstream[:DFL], bitstream[DFL:]



# ============================================================
#  BUILD 80-bit BBHEADER (ETSI §5.1.6)
# ============================================================

def build_bbheader(MATYPE1, MATYPE2, UPL, DFL, SYNC, SYNCd):
    """
    Correct DVB-S2 BBHEADER:
      8  bits – MATYPE1
      8  bits – MATYPE2
      16 bits – UPL
      16 bits – DFL
      8  bits – SYNC
      8  bits – SYNCd
      ---------------------
      72 bits total before CRC
      + 8-bit CRC = 80 bits
    """

    def _bits(value, width):
        return [int(b) for b in format(value & ((1 << width) - 1), f"0{width}b")]

    header_bits = []
    header_bits += _bits(MATYPE1, 8)
    header_bits += _bits(MATYPE2, 8)
    header_bits += _bits(UPL, 16)
    header_bits += _bits(DFL, 16)
    header_bits += _bits(SYNC, 8)
    header_bits += _bits(SYNCd, 8)
    header_bits += _bits(0, 8)  # reserved byte to reach 72 bits before CRC

    # VERIFY → 72 bits BEFORE CRC
    assert len(header_bits) == 72, f"Header before CRC should be 72 bits, got {len(header_bits)}"

    # Compute CRC-8 over 72 bits
    crc = dvbs2_crc8(np.array(header_bits, dtype=np.uint8))

    # Append 8 CRC bits
    header_bits += _bits(crc, 8)

    # VERIFY FINAL LENGTH = 80 bits
    assert len(header_bits) == 80, f"BBHEADER must be 80 bits, got {len(header_bits)}"

    return np.array(header_bits, dtype=np.uint8)



# ============================================================
#  DUMMY PLFRAME
# ============================================================

def generate_dummy_plframe(DFL):

    header = build_bbheader(
        MATYPE1=0,
        MATYPE2=0,
        UPL=0,
        DFL=DFL,
        SYNC=0,
        SYNCd=0
    )

    df = np.zeros(DFL, dtype=np.uint8)

    return np.concatenate([header, df])



# ============================================================
#  MAIN MERGER/SLICER PIPELINE
# ============================================================

def dvbs2_merger_slicer_pipeline():

    # ------------------------------------
    # USER INPUTS
    # ------------------------------------

    stream_type = input("Enter stream type (TS or GS): ").strip().upper()

    fecframe = input("Enter FECFRAME type (normal/short): ").strip().lower()


    Kbch = KBCH_NORMAL if fecframe == "normal" else KBCH_SHORT


    DFL = int(input(f"Enter DFL (max {Kbch-80}): "))

    if DFL > Kbch - 80:

        raise ValueError("DFL exceeds Kbch-80")


    UPL = 0

    if stream_type == "GS":

        UPL = int(input("Enter UPL (0 for continuous GS): "))



    # ------------------------------------
    # CREATE INPUT PACKETS
    # ------------------------------------

    data = "The quick brown fox jumps over the lazy dog"

    data_bits = convert_to_bits(data)


    next_sync = 0x47


    if stream_type == "TS":

        pkt = generate_ts_packet(data_bits, next_sync)

    else:

        pkt = generate_gs_packet(data_bits, UPL, next_sync)


    crc_val = dvbs2_crc8(pkt[8:])

    print("CRC-8 for next packet sync =", hex(crc_val))


    if stream_type == "TS":

        pkt2 = generate_ts_packet(data_bits, crc_val)

    else:

        pkt2 = generate_gs_packet(data_bits, UPL, crc_val)


    full_stream = np.concatenate([pkt, pkt2])



    # ------------------------------------
    # SLICING
    # ------------------------------------

    DF, leftover = slice_stream(full_stream, DFL)



    # ------------------------------------
    # SYNCd (TS only)
    # ------------------------------------

    SYNC = crc_val

    SYNCd = compute_sync_distance(DF) if stream_type == "TS" else 0



    # ------------------------------------
    # BUILD BBHEADER
    # ------------------------------------

    header = build_bbheader(
        MATYPE1 = 0x00,
        MATYPE2 = 0x00,
        UPL     = UPL,
        DFL     = DFL,
        SYNC    = SYNC,
        SYNCd   = SYNCd
    )


    PLFRAME = np.concatenate([header, DF])


    print("Generated PLFRAME:", len(PLFRAME), "bits")

    return PLFRAME
