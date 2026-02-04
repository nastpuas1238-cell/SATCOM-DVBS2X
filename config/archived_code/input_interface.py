import numpy as np
import pandas as pd

def convert_to_bits(data: str) -> np.ndarray:
    """Convert string to bits and return as numpy array."""
    bits = ''.join(format(ord(c), '08b') for c in data)
    return np.array([int(b) for b in bits], dtype=np.uint8)

def generate_ts_packet(data: str) -> np.ndarray:
    """Generate a Transport Stream (TS) packet of exactly 188 bytes (including sync byte)."""
    # Convert the string data to bits
    data_bits = convert_to_bits(data)
    
    # Total length should be 188 bytes, including the sync byte
    total_length = 188 * 8  # 188 bytes = 1504 bits (including sync byte)
    data_length = (188 - 1) * 8  # 1 byte for sync byte, the rest for the actual data
    
    if len(data_bits) < data_length:
        # Pad data with zeros if it's shorter
        data_bits = np.pad(data_bits, (0, data_length - len(data_bits)), mode='constant')
    else:
        # Truncate data if it's longer
        data_bits = data_bits[:data_length]
    
    # Add sync byte (0x47) at the beginning (1 byte)
    sync_byte = 0x47
    sync_bits = np.array([int(b) for b in format(sync_byte, '08b')], dtype=np.uint8)  # Sync byte as bits
    
    # Concatenate sync byte with data
    packet = np.concatenate([sync_bits, data_bits])
    
    return packet

def generate_gs_packet(data: str, upl: int) -> np.ndarray:
    """Generate a Generic Stream (GS) packet."""
    data_bits = convert_to_bits(data)
    
    # For Generic Stream, add sync byte (0x47) at the start of the packet
    sync_byte = 0x47
    sync_bits = np.array([int(b) for b in format(sync_byte, '08b')], dtype=np.uint8)
    
    # For GS, we use the UPL as the exact packet length
    if upl <= 64 * 1024:  # If UPL is less than or equal to 64 kbits
        # Create packetized stream of length UPL
        total_length = upl
        if len(data_bits) < total_length:
            # Pad data with zeros
            data_bits = np.pad(data_bits, (0, total_length - len(data_bits)), mode='constant')
        else:
            data_bits = data_bits[:total_length]
        
        # Concatenate sync byte with the data
        packet = np.concatenate([sync_bits, data_bits])
        return packet
    else:
        # For UPL > 64 kbits, treat as continuous stream
        print("Treating stream as continuous (UPL > 64 kbits)")
        # Here you could handle continuous stream, but for now, just add sync byte at the start
        packet = np.concatenate([sync_bits, data_bits])
        return packet

def save_to_excel(data: np.ndarray, filename: str):
    """Save the data to an Excel file."""
    df = pd.DataFrame(data, columns=['Bit'])
    df.to_excel(filename, index=False, engine='openpyxl')
    print(f"Data saved to {filename}")

def read_and_process_input():
    """Read stream type and frame type, then process the data."""
    # Example data from the user (can be replaced by actual data stream)
    data = "The quick brown fox jumps over the lazy dog"
    
    # Take user inputs for stream type
    stream_type = input("Enter stream type (TS or GS): ").strip()

    # If stream type is TS, no need for UPL input (fixed 188 bytes)
    if stream_type == 'TS':
        # Generate a TS packet (Transport Stream)
        bitstream = generate_ts_packet(data)
        filename = "TS_output.xlsx"
        save_to_excel(bitstream, filename)
    
    elif stream_type == 'GS':
        # For Generic Stream, ask for UPL
        upl = int(input("Enter UPL (packet length in bits) for GS (max 64kbits): ").strip())
        # Generate a GS packet (Generic Stream)
        bitstream = generate_gs_packet(data, upl)
        filename = "GS_output.xlsx"
        save_to_excel(bitstream, filename)
    
    else:
        print("Invalid stream type. Please enter either 'TS' or 'GS'.")
        return None
    
    # Return the generated bitstream for further operations
    return bitstream

# Run the function to read input and process the data
bitstream = read_and_process_input()

