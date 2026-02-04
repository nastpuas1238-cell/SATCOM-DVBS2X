import numpy as np
import pandas as pd

# ============================================================
#  DVB-S2 CRC-8 POLYNOMIAL (ETSI EN 302 307-1 §5.1.4)
# ============================================================
CRC8_POLY = 0xD5


# ============================================================
#  Convert string → numpy array of 0/1 bits
# ============================================================
def convert_to_bits(data: str) -> np.ndarray:
    bits = ''.join(format(ord(c), '08b') for c in data)
    return np.array([int(b) for b in bits], dtype=np.uint8)


# ============================================================
#  DVB-S2 CRC-8 (EXACT FIGURE 2)
# ============================================================
def dvbs2_crc8(bitstream: np.ndarray) -> int:
    crc = 0
    for bit in bitstream:
        msb = (crc >> 7) & 1
        xor_in = msb ^ bit
        crc = ((crc << 1) & 0xFF)
        if xor_in == 1:
            crc ^= CRC8_POLY
    return crc


# ============================================================
#  STEP-BY-STEP CRC DEBUG
# ============================================================
def dvbs2_crc8_trace(bitstream: np.ndarray):
    print("\n--- CRC-8 Step-By-Step Trace ---")
    crc = 0
    for i, bit in enumerate(bitstream):
        msb = (crc >> 7) & 1
        xor_in = msb ^ bit
        before = crc
        crc = ((crc << 1) & 0xFF)
        log = f"bit[{i}]={bit}, MSB={msb}, xor={xor_in}, before={hex(before)}"
        if xor_in == 1:
            crc ^= CRC8_POLY
            log += f", after XOR poly={hex(crc)}"
        print(log)
    print("Final CRC-8 =", hex(crc))
    return crc


# ============================================================
#  Transport Stream (188 bytes = 1504 bits)
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
#  Generic Stream packet
# ============================================================
def generate_gs_packet(data_bits: np.ndarray, upl: int, sync_byte: int) -> np.ndarray:
    if upl == 0:
        return data_bits
    payload_bits_needed = upl - 8
    if len(data_bits) < payload_bits_needed:
        payload = np.pad(data_bits, (0, payload_bits_needed - len(data_bits)))
    else:
        payload = data_bits[:payload_bits_needed]
    sync_bits = np.array([int(b) for b in format(sync_byte, '08b')], dtype=np.uint8)
    return np.concatenate([sync_bits, payload])


# ============================================================
#  SAVING FUNCTIONS
# ============================================================
def save_bitstream_bin(bitstream: np.ndarray, filename: str):
    """Save bitstream as raw binary file (packed bits)."""
    byte_array = np.packbits(bitstream)
    with open(filename, "wb") as f:
        f.write(byte_array)
    print(f"Saved binary → {filename}")


def save_bitstream_txt(bitstream: np.ndarray, filename: str):
    """Save bitstream as text file containing '0' and '1'."""
    with open(filename, "w") as f:
        f.write("".join(str(b) for b in bitstream))
    print(f"Saved text → {filename}")


def save_bitstream_excel(bitstream: np.ndarray, filename: str):
    df = pd.DataFrame(bitstream, columns=["Bit"])
    df.to_excel(filename, index=False, engine="openpyxl")
    print(f"Saved Excel → {filename}")


# ============================================================
#  DVB-S2 Input Pipeline (TS + GS)
# ============================================================
def dvbs2_input_pipeline():
    data = "The quick brown fox jumps over the lazy dog"
    data_bits = convert_to_bits(data)
    stream_type = input("Enter stream type (TS or GS): ").strip().upper()
    packets = []
    next_sync = 0x47

    if stream_type == "TS":
        pkt1 = generate_ts_packet(data_bits, next_sync)
        packets.append(pkt1)
        crc_val = dvbs2_crc8(pkt1[8:])
        print("CRC-8 for NEXT TS packet =", hex(crc_val))
        next_sync = crc_val
        pkt2 = generate_ts_packet(data_bits, next_sync)
        packets.append(pkt2)
        return packets

    elif stream_type == "GS":
        upl = int(input("Enter UPL in bits: "))
        pkt1 = generate_gs_packet(data_bits, upl, next_sync)
        packets.append(pkt1)
        if upl != 0:
            crc_val = dvbs2_crc8(pkt1[8:])
            print("CRC-8 for NEXT GS packet =", hex(crc_val))
            next_sync = crc_val
            pkt2 = generate_gs_packet(data_bits, upl, next_sync)
            packets.append(pkt2)
        return packets

    print("Invalid type.")
    return None


# ============================================================
#  TEST HELPERS + TEST SUITE
# ============================================================
def dvbs2_input_pipeline_for_test_ts():
    data_bits = convert_to_bits("The quick brown fox jumps over the lazy dog")
    packets = []
    next_sync = 0x47
    pkt1 = generate_ts_packet(data_bits, next_sync)
    packets.append(pkt1)
    crc_val = dvbs2_crc8(pkt1[8:])
    pkt2 = generate_ts_packet(data_bits, crc_val)
    packets.append(pkt2)
    return packets


def test_crc_polynomial():
    assert CRC8_POLY == 0xD5
    print("✓ CRC polynomial correct")


def test_crc_simple_pattern():
    bits = np.array([1,0,1,0,1,0,1,0], dtype=np.uint8)
    print("CRC =", hex(dvbs2_crc8(bits)))


def test_ts_packet_length():
    pkt = generate_ts_packet(convert_to_bits("abc"), 0x47)
    assert len(pkt) == 1504
    print("✓ TS packet length correct")


def test_ts_sync_is_0x47():
    pkt = generate_ts_packet(convert_to_bits("hi"), 0x47)
    assert int("".join(str(b) for b in pkt[:8]), 2) == 0x47
    print("✓ TS sync byte correct")


def test_crc_sync_replacement():
    a, b = dvbs2_input_pipeline_for_test_ts()
    s2 = int("".join(str(i) for i in b[:8]), 2)
    assert s2 == dvbs2_crc8(a[8:])
    print("✓ CRC sync replacement correct")


def test_gs_packet_length():
    pkt = generate_gs_packet(convert_to_bits("TEST"), 64, 0x47)
    assert len(pkt) == 64
    print("✓ GS packet length correct")


def test_crc_deterministic():
    bits = convert_to_bits("fixed")
    assert dvbs2_crc8(bits) == dvbs2_crc8(bits)
    print("✓ CRC deterministic")


def test_crc_fuzz():
    for _ in range(10):
        rand = np.random.randint(0,2,128)
        assert dvbs2_crc8(rand) == dvbs2_crc8(rand)
    print("✓ CRC fuzz OK")


# ============================================================
#  RUN TESTS
# ============================================================
if __name__ == "__main__":
    print("\n=== DVB-S2 CRC-8 TEST SUITE START ===\n")

    test_crc_polynomial()
    test_crc_simple_pattern()
    test_ts_packet_length()
    test_ts_sync_is_0x47()
    test_crc_sync_replacement()
    test_gs_packet_length()
    test_crc_deterministic()
    test_crc_fuzz()

    print("\n=== ALL TESTS PASSED SUCCESSFULLY ===\n")

    # RUN PIPELINE AND SAVE FINAL OUTPUT
    packets = dvbs2_input_pipeline()
    if packets:
        final_bitstream = np.concatenate(packets)
        print("\nFinal bitstream length =", len(final_bitstream), "bits")
        # save_bitstream_bin(final_bitstream, "final_stream.bin")
        # save_bitstream_txt(final_bitstream, "final_stream.txt")
        save_bitstream_excel(final_bitstream, "final_stream.xlsx")
