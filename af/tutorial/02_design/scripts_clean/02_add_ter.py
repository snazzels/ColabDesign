#!/usr/bin/env python3

import os
import argparse
import subprocess

def add_ter_lines(pdb_filename, output_filename):
    with open(pdb_filename, 'r') as pdb_file:
        lines = pdb_file.readlines()

    new_lines = []
    last_residue_id = None

    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            # Extract residue ID from the line
            current_residue_id = int(line[22:26].strip())

            if last_residue_id is not None and current_residue_id != last_residue_id:
                # If there is a jump in residue IDs, add a TER line
                if current_residue_id != last_residue_id + 1:
                    new_lines.append("TER\n")

            last_residue_id = current_residue_id
            new_lines.append(line)
        else:
            # Add lines that are not ATOM or HETATM as they are
            new_lines.append(line)

    # Ensure there is a TER line at the end if the last line was an ATOM/HETATM line
    if new_lines and not new_lines[-1].strip().startswith("TER"):
        new_lines.append("TER\n")

    with open(output_filename, 'w') as output_file:
        output_file.writelines(new_lines)

def run_pdb4amber(input_file, output_file):
    cmd = f"pdb4amber -i {input_file} -o {output_file}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"pdb4amber processed: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to run pdb4amber on {input_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Add TER lines to PDB files and clean them with pdb4amber.")
    parser.add_argument("input_files", nargs='+', help="The path to one or more PDB files to process")
    args = parser.parse_args()

    for input_file in args.input_files:
        if not os.path.isfile(input_file):
            print(f"File not found: {input_file}")
            continue

        base_name = os.path.splitext(input_file)[0]
        ter_output_file = f"{base_name}_ter.pdb"
        cleaned_output_file = f"{base_name}_cleaned.pdb"
        
        add_ter_lines(input_file, ter_output_file)
        run_pdb4amber(ter_output_file, cleaned_output_file)

if __name__ == "__main__":
    main()

