#!/home/niklashalbwedl/bin_custom/micromamba/envs/design/bin/python3

import os
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.shared.utils import copy_dict
from colabdesign.af.alphafold.common import residue_constants
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

def add_cyclic_offset(self, bug_fix=True):
    """
    Add cyclic offset to connect N and C terminus.

    Parameters:
    self : object
        An object containing the necessary attributes and methods.
    bug_fix : bool, optional
        If True, apply a fix for a bug related to offset calculation (default is True).
    """
    def cyclic_offset(L):
        i = np.arange(L)
        ij = np.stack([i, i + L], -1)
        offset = i[:, None] - i[None, :]
        c_offset = np.abs(ij[:, None, :, None] - ij[None, :, None, :]).min((2, 3))
        if bug_fix:
            a = c_offset < np.abs(offset)
            c_offset[a] = -c_offset[a]
        return c_offset * np.sign(offset)

    idx = self._inputs["residue_index"]
    offset = np.array(idx[:, None] - idx[None, :])

    if self.protocol == "binder":
        c_offset = cyclic_offset(self._binder_len)
        offset[self._target_len:, self._target_len:] = c_offset

    self._inputs["offset"] = offset


class PeptideLoss:
    def __init__(self, n_pep_res=7, hotspot_res=None):
        self.n_pep_res = n_pep_res

        if hotspot_res is None:
            raise ValueError("Please provide hotspot residues.")

        hotspot_res_split = hotspot_res.split(',')
        
        # Extract the numbers from each element
        self.hotspot_res = [int(''.join(filter(str.isdigit, element))) for element in hotspot_res_split]

        # Convert hotspot_res to zero-indexed
        self.hotspot_res = [(res - 1) for res in self.hotspot_res]


    def smooth_sigmoid(self, x, k=10):
        """
        Smooth sigmoid-like function using tanh to avoid overflow and provide a continuous function.

        Parameters:
        x : array_like
            Input array.
        k : float, optional
            Controls the steepness of the transition (default is 10).

        Returns:
        jnp.ndarray
            Output array after applying the smooth sigmoid function.
        """
        return 0.5 * (1 + jnp.tanh(k * x))



    def compute_dihedral(self,p0, p1, p2, p3):
        """
        Compute the dihedral angle between four atoms.

        Parameters:
        p0, p1, p2, p3 : array_like
            Four points in space defining the dihedral angle.

        Returns:
        jnp.ndarray
            Dihedral angle in radians.
        """
        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        # Normalize b1 so that it does not influence magnitude of vector
        b1 /= jnp.linalg.norm(b1)

        # v = projection of b0 onto plane perpendicular to b1
        # w = projection of b2 onto plane perpendicular to b1
        v = b0 - jnp.dot(b0, b1) * b1
        w = b2 - jnp.dot(b2, b1) * b1

        # x = dot product of normalized projections
        x = jnp.dot(v, w)
        # y = magnitude of cross product of projections
        y = jnp.dot(jnp.cross(b1, v), w)

        return jnp.arctan2(y, x)

    def compute_cis_penalty(self, angle_rad):
        """
        Compute a penalty for an omega angle based on its deviation from the -50 to 50 degree range.

        Parameters:
        angle_rad : float
            Omega angle in radians.

        Returns:
        jnp.ndarray
            Penalty value based on the omega angle deviation.
        """
        angle_deg = jnp.degrees(angle_rad)
        angle_deg = (angle_deg + 180) % 360 - 180  # Wrap angle to [-180, 180]

        dist_from_bound = jnp.minimum(jnp.abs(angle_deg - 50), jnp.abs(angle_deg + 50))
        penalty = self.smooth_sigmoid(-dist_from_bound + 5)  # Adjust k for a sharper transition
        return jnp.where(jnp.abs(angle_deg) <= 50, 1.0, penalty)

    def cis_loss(self, inputs, outputs):
        """
        Compute the loss based on cis conformation of omega angles in the cyclic peptide.

        Parameters:
        inputs : dict
            Dictionary containing the input data.
        outputs : dict
            Dictionary containing the output data, including final atom positions.

        Returns:
        dict
            Dictionary containing the 'cis' loss value.
        """
        positions = outputs["structure_module"]["final_atom_positions"]
        CA_atoms = positions[:, residue_constants.atom_order["CA"]]
        C_atoms = positions[:, residue_constants.atom_order["C"]]
        N_atoms = positions[:, residue_constants.atom_order["N"]]

        n_residues = len(C_atoms)
        if n_residues < 2:
            return {"cis": 0.0}  # No omega angles to calculate if fewer than 2 residues

        omega_angles = []


        for i in range(max(0, n_residues - self.n_pep_res - 1), n_residues - 1):
            p0 = CA_atoms[i]
            p1 = C_atoms[i]
            p2 = N_atoms[i + 1]
            p3 = CA_atoms[i + 1]
            omega_angle = self.compute_dihedral(p0, p1, p2, p3)
            omega_angles.append(omega_angle)

        omega_angles = jnp.array(omega_angles)
        penalties = jnp.array([self.compute_cis_penalty(angle) for angle in omega_angles])
        penalty = jnp.sum(penalties)

        return {"cis": penalty / len(omega_angles) if len(omega_angles) > 0 else 0.0}

    def compute_geometric_center(self, inputs, outputs, residue_indices):
        """
        Compute the geometric center of specified residues.

        Parameters:
        inputs : dict
            Dictionary containing the input data.
        outputs : dict
            Dictionary containing the output data, including final atom positions.
        residue_indices : list or np.ndarray
            List or array of residue indices for which to compute the geometric center.

        Returns:
        jnp.ndarray
            Geometric center of the specified residues.
        """
        positions = outputs["structure_module"]["final_atom_positions"]
        CA_atoms = positions[:, residue_constants.atom_order["CA"]]

        if residue_indices is None or not isinstance(residue_indices, (list, np.ndarray)):
            raise ValueError("residue_indices must be a list or numpy array of residue indices.")
        
        if any(idx >= len(CA_atoms) or idx < 0 for idx in residue_indices):
            raise ValueError("One or more residue indices are out of bounds.")

        selected_CA_atoms = CA_atoms[jnp.array(residue_indices)]
        centroid = jnp.mean(selected_CA_atoms, axis=0)

        return centroid

    def com_loss(self, inputs, outputs):
        """
        Compute the center of mass (com) loss, i.e., the distance between the center of the
        peptide and the PPI binding pocket.

        Parameters:
        inputs : dict
            Dictionary containing the input data.
        outputs : dict
            Dictionary containing the output data, including final atom positions.

        Returns:
        dict
            Dictionary containing the 'com_loss' value.
        """
        CA_atoms = outputs["structure_module"]["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        n_residues = len(CA_atoms)

        peptide_res = list(range(max(0, n_residues - self.n_pep_res), n_residues))
        center_peptide = self.compute_geometric_center(inputs, outputs, peptide_res)
        center_hotspot = self.compute_geometric_center(inputs, outputs, self.hotspot_res)
        distance = jnp.linalg.norm(center_peptide - center_hotspot)

        return {"com_loss": distance}

