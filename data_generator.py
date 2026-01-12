bitstream = "0111010101101101011000010110100101110010"

filename = "bits_single_column.csv"

total_bits = 0

with open(filename, "w") as f:
    for _ in range(1_000_000):
        for b in bitstream:
            f.write(b + "\n")
            total_bits += 1

print("CSV created:", filename)
print("Total bits written:", total_bits)
