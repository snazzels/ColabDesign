#!/home/niklashalbwedl/bin_custom/micromamba/envs/dihedral/bin/python3
import warnings
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import os
import re
import shutil  # For moving files

def get_chain_b_sequence(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('PDB', pdb_file)
    sequence = ""
    first_model = next(structure.get_models())
    for chain in first_model:
        if chain.id == 'C':  # Assuming chain C is the peptide
            for residue in chain:
                if residue.id[0] == ' ':  # Exclude heteroatoms
                    sequence += seq1(residue.get_resname())
    return sequence

def count_mutations(seq1, seq2):
    mutations = 0
    for a, b in zip(seq1, seq2):
        if a != b:
            mutations += 1
    return mutations

def print_chain_b_sequences(pdb_files):
    previous_sequence = None
    cys_dir = "CYS"
    if not os.path.exists(cys_dir):
        os.makedirs(cys_dir)  # Create the CYS directory if it doesn't exist

    for pdb_file in pdb_files:
        if not pdb_file.lower().endswith(".pdb"):
            continue
        sequence = get_chain_b_sequence(pdb_file)
        
        # Check if the sequence contains 'C' or 'X'
        if 'C' in sequence or 'X' in sequence:
            # Move the PDB file to the CYS directory
            shutil.move(pdb_file, os.path.join(cys_dir, pdb_file))
            print(f"Moved {pdb_file} to {cys_dir} due to presence of 'C' or 'X' in the sequence.")

        if previous_sequence is not None:
            mutations = count_mutations(previous_sequence, sequence)
            print(f"Sequence for chain B in {pdb_file}: {sequence} (Mutations: {mutations})")
        else:
            print(f"Sequence for chain B in {pdb_file}: {sequence} (Mutations: 0)")
        
        previous_sequence = sequence

# Assuming all your PDB files are in the current directory
pdb_files = [f for f in os.listdir('.') if os.path.isfile(f) and f.lower().endswith('.pdb')]

# Extract numeric part of the filename and sort files
pdb_files.sort(key=lambda f: int(re.search(r'model_step_(\d+)\_cleaned.pdb', f).group(1)))

print_chain_b_sequences(pdb_files)

