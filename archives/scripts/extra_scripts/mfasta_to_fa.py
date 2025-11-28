# Script to turn a clean mfasta into a fa
# Useful for generating the embeddings

def convert_mfasta_to_fa(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        sequence = ""
        header = None

        for line in infile:
            line = line.strip()
            if line.startswith(">"):  # New header line
                if header and sequence:
                    outfile.write(f"{header}\n{sequence}\n")
                header = line  # Save the new header
                sequence = ""  # Reset sequence
            else:  # Sequence line
                sequence += line.replace("-", "")  # Remove gaps

        # Write the last sequence
        if header and sequence:
            outfile.write(f"{header}\n{sequence}\n")

# Example usage
input_file = "examples/thioredoxins/thioredoxins.mfasta"
output_file = "thioredoxins.fa"
convert_mfasta_to_fa(input_file, output_file)
