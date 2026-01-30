import numpy as np
from common.bit_interleaver import dvbs2_bit_interleave, dvbs2_bit_deinterleave, BITS_PER_SYMBOL


def test_interleaver_roundtrip():
    for mod, m in BITS_PER_SYMBOL.items():
        N = m * 16
        x = np.random.randint(0, 2, size=N, dtype=np.uint8)
        y = dvbs2_bit_interleave(x, mod)
        z = dvbs2_bit_deinterleave(y, mod)
        assert len(y) == N, f"Length changed for {mod}"
        assert np.array_equal(x, z), f"Roundtrip failed for {mod}"
    print("test_interleaver_roundtrip: PASSED")


def test_interleaver_permutation():
    # Ensure interleaver is a permutation of input bits (same multiset)
    for mod, m in BITS_PER_SYMBOL.items():
        N = m * 32
        x = np.random.randint(0, 2, size=N, dtype=np.uint8)
        y = dvbs2_bit_interleave(x, mod)
        # counts of ones should match
        assert int(np.sum(x)) == int(np.sum(y)), f"Bit counts differ for {mod}"
    print("test_interleaver_permutation: PASSED")


def test_integration_with_ldpc_shape():
    # Simulate an LDPC codeword length for normal frames (e.g., 64800)
    # Test interleaver accepts lengths divisible by bits-per-symbol and rejects others
    ldpc_len = 64800
    for mod, m in BITS_PER_SYMBOL.items():
        bits = np.random.randint(0, 2, size=ldpc_len, dtype=np.uint8)
        if ldpc_len % m == 0:
            y = dvbs2_bit_interleave(bits, mod)
            assert len(y) == ldpc_len
        else:
            try:
                dvbs2_bit_interleave(bits, mod)
                raise AssertionError("Expected ValueError for incompatible length")
            except ValueError:
                pass
    print("test_integration_with_ldpc_shape: PASSED")


# ============================================================
# Constellation mapper tests
# ============================================================
from constellation_mapper import dvbs2_constellation_map, map_qpsk, map_8psk


def test_qpsk_known_mapping():
    # MSB-first groups: 00,01,11,10
    bits = np.array([0,0, 0,1, 1,1, 1,0], dtype=np.uint8)
    syms = map_qpsk(bits)
    expected = np.array([1+1j, 1-1j, -1-1j, -1+1j], dtype=np.complex128) / np.sqrt(2.0)
    assert np.allclose(syms, expected)
    print("test_qpsk_known_mapping: PASSED")


def test_qpsk_power():
    bits = np.random.randint(0,2, size=200, dtype=np.uint8)
    syms = dvbs2_constellation_map(bits, 'QPSK')
    avg_power = np.mean(np.abs(syms)**2)
    assert np.isclose(avg_power, 1.0, atol=1e-6)
    print("test_qpsk_power: PASSED")


def test_8psk_unit_magnitude():
    bits = np.arange(24) % 2
    syms = map_8psk(bits, use_lut=True)
    mags = np.abs(syms)
    assert np.allclose(mags, 1.0)
    print("test_8psk_unit_magnitude: PASSED")


if __name__ == '__main__':
    test_interleaver_roundtrip()
    test_interleaver_permutation()
    test_integration_with_ldpc_shape()
    print("All interleaver tests PASSED")

    # Run constellation mapper tests
    test_qpsk_known_mapping()
    test_qpsk_power()
    test_8psk_unit_magnitude()
    print("All constellation mapper tests PASSED")
