#!/usr/bin/env python3

import os
import argparse

def assign_chain_identifiers(pdb_file, output_file):
    current_chain = 'A'  # Start with chain A
    chain_switched = False

    with open(pdb_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as out:
        for line in lines:
            if line.startswith("ATOM"):
                # Insert the chain identifier directly into the correct column (21st index in 0-based, 22nd in 1-based)
                line = line[:21] + current_chain + line[22:]
                out.write(line)
            elif line.startswith("TER"):
                # Write the TER line with the current chain identifier and switch chain
                ter_line = line[:21] + current_chain + line[22:]
                out.write(ter_line)
                if not chain_switched:
                    current_chain = 'B'
                    chain_switched = True
                else:
                    current_chain = chr(ord(current_chain) + 1)  # Increment to the next chain identifier
            else:
                # Write all other lines unchanged
                out.write(line)

def process_cleaned_pdb_files(input_files, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for input_file in input_files:
        base_name = os.path.basename(input_file)
        output_file = os.path.join(output_dir, base_name)
        assign_chain_identifiers(input_file, output_file)
        print(f"Processed {input_file} and saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Assign chain identifiers to cleaned PDB files.")
    parser.add_argument("input_files", nargs='+', help="The path to one or more cleaned PDB files to process")
    parser.add_argument("-o", "--output_dir", default="cleaned_top", help="The directory to save processed PDB files")
    args = parser.parse_args()

    process_cleaned_pdb_files(args.input_files, args.output_dir)

if __name__ == "__main__":
    main()

