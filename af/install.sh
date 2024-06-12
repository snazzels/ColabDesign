#!/usr/bin/bash


# Initialize shell for micromamba
eval "$(micromamba shell hook --shell bash)"

# Create and activate environment
micromamba create -n test_colab
micromamba activate test_colab

# Install Python 3.12 and JAX
micromamba install python=3.12
micromamba install jax=0.4.23 --channel conda-forge

# Install colab design
pip install git+https://github.com/snazzels/ColabDesign.git

# Download alphafold weights.
mkdir params
curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar | tar x -C params


# Ensure compatibility of Python, SciPy, and JAX versions
# https://github.com/google/jax/issues/20565

