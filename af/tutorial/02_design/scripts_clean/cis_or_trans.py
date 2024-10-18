#!/home/niklashalbwedl/bin_custom/micromamba/envs/dihedral/bin/python3
import mdtraj as md
import numpy as np
import os
import sys
import glob
import shutil

def is_cis_or_trans(angle):
    """Determine if the peptide bond is cis or trans based on the omega dihedral angle."""
    if 140 <= angle <= 210 or -210 <= angle <= -140:
        return 'trans'
    else:
        return 'cis'

def process_pdb(file_path):
    # Load the cyclic peptide structure
    traj = md.load(file_path)
    
    # Identify the C, CA, and N atom indices for chain B
    topology = traj.topology

    C_atoms = []
    CA_atoms = []
    N_atoms = []
    residues = []

    # Identify chain B (adjust if chain B has a different index)
    chain_B = topology.chain(2)

    for res in chain_B.residues:
        if res.is_protein:
            try:
                C_atoms.append(res.atom('C').index)
                CA_atoms.append(res.atom('CA').index)
                N_atoms.append(res.atom('N').index)
                residues.append(res)
            except KeyError:
                # Skip if required atoms are not found (which shouldn't happen in a well-formed peptide)
                pass

    # Identify the CA-C-N-CA dihedral indices for chain B
    n_residues = len(C_atoms)
    dihedral_indices = []

    for i in range(n_residues - 1):  # only consider consecutive residues within the chain
        dihedral_indices.append([CA_atoms[i], C_atoms[i], N_atoms[i + 1], CA_atoms[i + 1]])

    # Measure the omega dihedral angles associated with these peptide bonds
    dihedral_indices = np.array(dihedral_indices)
    omega_angles = md.compute_dihedrals(traj, dihedral_indices)

    # Determine cis/trans status for each peptide bond and count cis bonds
    cis_count = 0
    for angle_rad in omega_angles[0]:
        angle_deg = np.degrees(angle_rad)
        status = is_cis_or_trans(angle_deg)
        if status == 'cis':
            cis_count += 1

    # Print the results
    print(f"{file_path}, Number of cis bonds: {cis_count}")

    # Move the file based on the cis count
    if cis_count == 0:
        target_dir = os.path.join(os.path.dirname(file_path), 'trans')
    else:
        target_dir = os.path.join(os.path.dirname(file_path), 'cis')
    
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Move the file
    shutil.move(file_path, os.path.join(target_dir, os.path.basename(file_path)))

def main():
    pdb_files = glob.glob('*.pdb')

    for pdb_file in pdb_files:
        process_pdb(pdb_file)

if __name__ == "__main__":
    main()

