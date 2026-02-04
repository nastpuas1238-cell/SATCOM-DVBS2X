import numpy as np

# ============================================================
#  DVB-S2 CONSTANTS (ETSI EN 302 307 V1.3.1)
# ============================================================

CRC8_POLY = 0xD5  # g(x)=x^8+x^7+x^6+x^4+x^2+1  (clause 5.1.4)

KBCH_NORMAL = 64800
KBCH_SHORT  = 16200

SYNC_TS = 0x47  # MPEG-TS sync byte



def dvbs2_crc8(bitstream_bits: np.ndarray) -> int:
    crc = 0
    for bit in bitstream_bits:
        msb = (crc >> 7) & 1
        xor_in = msb ^ int(bit)
        crc = ((crc << 1) & 0xFF)
        if xor_in:
            crc ^= CRC8_POLY
    return crc



def build_matype1(ts_gs_bits: int, sis: int, ccm: int, issyi: int, npd: int, ro_bits: int) -> int:
    """
    ts_gs_bits:
      0b11 = Transport Stream
      0b00 = Generic Packetized
      0b01 = Generic Continuous
    sis: 1=single, 0=multiple
    ccm: 1=CCM, 0=ACM
    issyi: 1 active, 0 not active
    npd: 1 active, 0 not active
    ro_bits: 0b00=0.35, 0b01=0.25, 0b10=0.20, 0b11 reserved
    """
    v = 0
    v |= (ts_gs_bits & 0b11) << 6
    v |= (sis & 0b1) << 5
    v |= (ccm & 0b1) << 4
    v |= (issyi & 0b1) << 3
    v |= (npd & 0b1) << 2
    v |= (ro_bits & 0b11)
    return v

def ro_to_bits(alpha: float) -> int:
    if abs(alpha - 0.35) < 1e-9:
        return 0b00
    if abs(alpha - 0.25) < 1e-9:
        return 0b01
    if abs(alpha - 0.20) < 1e-9:
        return 0b10
    raise ValueError("RO alpha must be one of: 0.35, 0.25, 0.20")

# ============================================================
#  BBHEADER builder (clause 5.1.6)
#  10 bytes total = 80 bits:
#   MATYPE-1 (8)
#   MATYPE-2 (8)
#   UPL (16)
#   DFL (16)
#   SYNC (8)
#   SYNCD (16)
#   CRC-8 (8) computed over first 9 bytes (72 bits)
# ============================================================

def build_bbheader(matype1: int, matype2: int, upl: int, dfl: int, sync: int, syncd: int) -> np.ndarray:
    def bits(value: int, width: int):
        return [int(b) for b in format(value & ((1 << width) - 1), f"0{width}b")]

    header72 = []
    header72 += bits(matype1, 8)
    header72 += bits(matype2, 8)
    header72 += bits(upl, 16)
    header72 += bits(dfl, 16)
    header72 += bits(sync, 8)
    header72 += bits(syncd, 16)

    assert len(header72) == 72, f"BBHEADER first 9 bytes must be 72 bits, got {len(header72)}"

    crc = dvbs2_crc8(np.array(header72, dtype=np.uint8))
    full = header72 + bits(crc, 8)

    assert len(full) == 80, f"BBHEADER must be 80 bits, got {len(full)}"
    return np.array(full, dtype=np.uint8)

# ============================================================
#  bits.bin loader (memmap, no RAM copy)
# ============================================================

def load_bits_bin(path="bits.bin") -> np.memmap:
    bits = np.memmap(path, dtype=np.uint8, mode="r")
    # Optional strict sanity check (fast sample, not full scan)
    sample = bits[:1000]
    if not np.all((sample == 0) | (sample == 1)):
        raise ValueError("bits.bin must contain only 0/1 uint8 values.")
    return bits

# ============================================================
#  CRC-8 ENCODER OUTPUT STREAM FOR PACKETIZED INPUTS (clause 5.1.4)
#  For packetized streams (UPL != 0):
#   - treat each UP as: [sync_byte(8 bits) + payload(UPL-8 bits)]
#   - compute CRC over payload only
#   - CRC replaces sync-byte of the FOLLOWING UP
#  For continuous streams (UPL == 0): pass through unchanged
# ============================================================

class PacketizedCrc8Stream:
    """
    Provides a sequential-read view of the MODE ADAPTER output stream after CRC-8 encoding.
    It reads from an input bitstream that contains original sync bytes at the start of each UP.
    """

    def __init__(self, in_bits: np.ndarray, upl_bits: int):
        if upl_bits <= 0:
            raise ValueError("PacketizedCrc8Stream requires UPL > 0")
        if upl_bits < 8:
            raise ValueError("UPL must be >= 8 bits for packetized streams")

        self.in_bits = in_bits
        self.upl = upl_bits
        self.ptr = 0  # bit pointer into input
        self.prev_crc = None  # CRC computed from previous UP payload
        self.global_out_pos = 0  # bit position in the output stream (after CRC-8 replacement)

    def _read_exact(self, n: int) -> np.ndarray:
        end = self.ptr + n
        if end > len(self.in_bits):
            # Not enough data -> pad zeros (ETSI allows dummy if no data; we pad for continuity here)
            chunk = self.in_bits[self.ptr:]
            pad = np.zeros(end - len(self.in_bits), dtype=np.uint8)
            self.ptr = len(self.in_bits)
            return np.concatenate([chunk, pad])
        chunk = np.array(self.in_bits[self.ptr:end], dtype=np.uint8, copy=False)
        self.ptr = end
        return chunk

    def read_bits(self, n: int) -> np.ndarray:
        """
        Read n bits from the CRC-8-processed stream (MODE ADAPTER output).
        """
        out = np.zeros(n, dtype=np.uint8)
        out_i = 0

        while out_i < n:
            # We produce output in UP-sized chunks
            # Build the next UP output if we are aligned to a UP boundary in output stream
            # Output UP structure remains UPL bits: [CRC_or_sync (8) + payload (UPL-8)]
            # The CRC placed in first byte of UP_k is CRC(payload of UP_{k-1}).
            # For the very first UP, keep original sync byte (no previous CRC exists).

            # Determine position inside a UP in output stream
            pos_in_up = self.global_out_pos % self.upl
            remaining_in_up = self.upl - pos_in_up
            take = min(n - out_i, remaining_in_up)

            # If we are at the start of a UP in output stream, we need to ensure we have
            # current UP payload available and compute its CRC to be used for NEXT UP.
            if pos_in_up == 0:
                # Read original UP from input: sync + payload
                up_in = self._read_exact(self.upl)
                up_sync_in = up_in[:8]
                up_payload = up_in[8:]

                # Decide output first byte for this UP
                if self.prev_crc is None:
                    # First UP: unchanged sync-byte (ETSI implies "following UP" gets replaced)
                    up_sync_out = up_sync_in
                else:
                    up_sync_out = np.array([int(b) for b in format(self.prev_crc, "08b")], dtype=np.uint8)

                # Compute CRC for this UP payload (will replace next UP sync)
                self.prev_crc = dvbs2_crc8(up_payload)

                # Cache the full output UP bits for this UP
                self._current_up_out = np.concatenate([up_sync_out, up_payload])

            # Copy the requested slice from the current output UP
            start = pos_in_up
            out[out_i:out_i + take] = self._current_up_out[start:start + take]

            out_i += take
            self.global_out_pos += take

        return out

# ============================================================
#  SYNCD computation (clauses 5.1.5 / 5.1.6)
#  For packetized TS/GS:
#    SYNCD = distance (bits) from start of DATA FIELD to the first UP start in that DATA FIELD
#    where "UP start" means first bit of CRC-8 (i.e., first byte of UP after replacement)
#    If no UP starts in DATA FIELD => SYNCD = 65535
#  For continuous GS:
#    SYNCD reserved (0..FFFF)
# ============================================================

def compute_syncd_packetized(df_start_global_pos: int, dfl: int, upl: int) -> int:
    # distance to next UP boundary in the MODE ADAPTER output stream
    mod = df_start_global_pos % upl
    dist = 0 if mod == 0 else (upl - mod)

    if dist >= dfl:
        return 0xFFFF  # 65535: no UP starts inside DATA FIELD
    return dist

# ============================================================
#  MAIN ETSI-COMPLIANT PIPELINE USING bits.bin
# ============================================================

def dvbs2_bbframe_generator_from_bits_bin(
    bits_path="bits.bin",
    max_frames=10
):
    # -----------------------------
    # USER INPUTS
    # -----------------------------
    stream_type = input("Enter stream type (TS or GS): ").strip().upper()
    fecframe    = input("Enter FECFRAME type (normal/short): ").strip().lower()
    alpha       = float(input("Enter roll-off alpha (0.35 / 0.25 / 0.20): ").strip())

    Kbch = KBCH_NORMAL if fecframe == "normal" else KBCH_SHORT
    DFL = int(input(f"Enter DFL (0..{Kbch-80}): "))
    if not (0 <= DFL <= Kbch - 80):
        raise ValueError("DFL must satisfy 0 <= DFL <= Kbch-80 (clause 5.1.5).")

    # For strict TS: UPL = 188*8 (clause 5.1.6 example)
    # For GS: user supplies UPL; UPL=0 means continuous generic stream (clause 5.1.4).
    if stream_type == "TS":
        UPL = 188 * 8
        SYNC = SYNC_TS
        ts_gs_bits = 0b11  # TS
    elif stream_type == "GS":
        UPL = int(input("Enter UPL in bits (0 for continuous GS): ").strip())
        if not (0 <= UPL <= 65535):
            raise ValueError("UPL must be in range 0..65535 (clause 5.1.6).")
        if UPL == 0:
            ts_gs_bits = 0b01  # Generic continuous
        else:
            ts_gs_bits = 0b00  # Generic packetized

        # SYNC rules (clause 5.1.6):
        # - packetized GS: SYNC is copy of user packet sync byte (or 0 if none)
        # - continuous GS: SYNC in 00..B8 reserved, B9..FF private.
        # Here: ask user explicitly to stay ETSI-correct.
        SYNC = int(input("Enter SYNC byte in hex (e.g., 47 or 00): ").strip(), 16) & 0xFF
    else:
        raise ValueError("stream_type must be TS or GS")

    # MATYPE-1 fields: assume SIS, CCM, ISSYI=0, NPD per user (TS only meaningful), RO as input
    sis = 1
    ccm = 1
    issyi = 0

    if stream_type == "TS":
        npd = int(input("Null-packet deletion active? (0/1): ").strip())
    else:
        npd = 0

    ro_bits = ro_to_bits(alpha)
    MATYPE1 = build_matype1(ts_gs_bits, sis, ccm, issyi, npd, ro_bits)

    # MATYPE-2: reserved for SIS; ISI for MIS
    MATYPE2 = 0x00

    # -----------------------------
    # LOAD INPUT BITS
    # -----------------------------
    in_bits = load_bits_bin(bits_path)

    # -----------------------------
    # CREATE MODE ADAPTER OUTPUT STREAM VIEW
    # -----------------------------
    if UPL == 0:
        # continuous GS: pass through unchanged
        mode_adapter_stream = in_bits
        stream_ptr = 0
        df_global_pos = 0  # global position in mode adapter output
        packetized = False
    else:
        # packetized: CRC-8 encoder modifies sync bytes of following UPs
        crc_stream = PacketizedCrc8Stream(in_bits, UPL)
        df_global_pos = 0
        packetized = True

    frames = 0
    while frames < max_frames:
        # -----------------------------
        # READ DFL BITS AS DATA FIELD (Merger/Slicer clause 5.1.5)
        # -----------------------------
        if DFL == 0:
            DF = np.array([], dtype=np.uint8)
        else:
            if not packetized:
                # continuous: slice directly from memmap
                end = stream_ptr + DFL
                if end <= len(mode_adapter_stream):
                    DF = np.array(mode_adapter_stream[stream_ptr:end], dtype=np.uint8, copy=False)
                else:
                    # Not enough bits available -> pad zeros (dummy-frame handling is later in PL framing;
                    # here we keep a well-defined BBFRAME output)
                    tail = mode_adapter_stream[stream_ptr:]
                    pad = np.zeros(end - len(mode_adapter_stream), dtype=np.uint8)
                    DF = np.concatenate([tail, pad])
                stream_ptr = min(end, len(mode_adapter_stream))
            else:
                DF = crc_stream.read_bits(DFL)

        # -----------------------------
        # SYNCD (clause 5.1.6)
        # -----------------------------
        if packetized:
            SYNCD = compute_syncd_packetized(df_start_global_pos=df_global_pos, dfl=DFL, upl=UPL)
        else:
            # reserved for continuous GS
            SYNCD = 0x0000

        # Update global position in MODE ADAPTER output stream
        df_global_pos += DFL

        # -----------------------------
        # BUILD BBHEADER (clause 5.1.6)
        # -----------------------------
        BBHEADER = build_bbheader(
            matype1=MATYPE1,
            matype2=MATYPE2,
            upl=UPL,
            dfl=DFL,
            sync=SYNC,
            syncd=SYNCD
        )

        BBFRAME = np.concatenate([BBHEADER, DF])
        frames += 1

        print(f"BBFRAME {frames}: {len(BBFRAME)} bits (BBHEADER=80 + DF={DFL}) | SYNCD={SYNCD}")

    return

# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    dvbs2_bbframe_generator_from_bits_bin(
        bits_path="bits.bin",
        max_frames=10
    )

