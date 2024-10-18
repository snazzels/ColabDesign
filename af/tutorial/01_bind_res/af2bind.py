#! /home/niklashalbwedl/bin_custom/micromamba/envs/design/bin/python3 -u

import json
import os
from colabdesign import mk_afdesign_model, clear_mem
import numpy as np

from colabdesign.af.alphafold.common import residue_constants
import pandas as pd
import jax, pickle
import jax.numpy as jnp
from scipy.special import expit as sigmoid
# 
import py3Dmol
import matplotlib.pyplot as plt
from scipy.special import softmax
import copy
from colabdesign.shared.protein import renum_pdb_str
from colabdesign.af.alphafold.common import protein


# Load parameters from the JSON configuration file
config_file = "config.json"
with open(config_file, "r") as f:
    config = json.load(f)

# Extract parameters from the JSON file
params_dir = config["params_dir"]
pdb_filename = config["pdb"]
target_chain = config["target_chain"]


aa_order = {v:k for k,v in residue_constants.restype_order.items()}
 
def af2bind(outputs, mask_sidechains=True, seed=0):
  pair_A = outputs["representations"]["pair"][:-20,-20:]
  pair_B = outputs["representations"]["pair"][-20:,:-20].swapaxes(0,1)
  pair_A = pair_A.reshape(pair_A.shape[0],-1)
  pair_B = pair_B.reshape(pair_B.shape[0],-1)
  x = np.concatenate([pair_A,pair_B],-1)

  # get params
  if mask_sidechains:
    model_type = f"split_nosc_pair_A_split_nosc_pair_B_{seed}"
  else:
    model_type = f"split_pair_A_split_pair_B_{seed}"
  with open(f"/home/niklashalbwedl/henmount/apps/af2bind_params/attempt_7_2k_lam0-03/{model_type}.pickle","rb") as handle:
    params_ = pickle.load(handle)
  params_ = dict(**params_["~"], **params_["linear"])
  p = jax.tree_map(lambda x:np.asarray(x), params_)

  # get predictions
  x = (x - p["mean"]) / p["std"]
  x = (x * p["w"][:,0]) + (p["b"] / x.shape[-1])
  p_bind_aa = x.reshape(x.shape[0],2,20,-1).sum((1,3))
  p_bind = sigmoid(p_bind_aa.sum(-1))
  return {"p_bind":p_bind, "p_bind_aa":p_bind_aa}



#@title **Run AF2BIND** 🔬
mask_sidechains = True
mask_sequence = False


clear_mem()
af_model = mk_afdesign_model(protocol="binder", debug=True, data_dir=params_dir)
af_model.prep_inputs(pdb_filename=pdb_filename,
                     chain=target_chain,
                     binder_len=20,
                     rm_target_sc=mask_sidechains,
                     rm_target_seq=mask_sequence)

# split
r_idx = af_model._inputs["residue_index"][-20] + (1 + np.arange(20)) * 50
af_model._inputs["residue_index"][-20:] = r_idx.flatten()

af_model.set_seq("ACDEFGHIKLMNPQRSTVWY")
af_model.predict(verbose=False)

o = af2bind(af_model.aux["debug"]["outputs"],
            mask_sidechains=mask_sidechains)
pred_bind = o["p_bind"].copy()
pred_bind_aa = o["p_bind_aa"].copy()

#######################################################
labels = ["chain","resi","resn","p(bind)"]
data = []
for i in range(af_model._target_len):
  c = af_model._pdb["idx"]["chain"][i]
  r = af_model._pdb["idx"]["residue"][i]
  a = aa_order.get(af_model._pdb["batch"]["aatype"][i],"X")
  p = pred_bind[i]
  data.append([c,r,a,p])

df = pd.DataFrame(data, columns=labels)
df.to_csv('results.csv')

top_n = 15
top_n_idx = pred_bind.argsort()[::-1][:15]
pymol_cmd="resid "
for n,i in enumerate(top_n_idx):
  p = pred_bind[i]
  c = af_model._pdb["idx"]["chain"][i]
  r = af_model._pdb["idx"]["residue"][i]
  pymol_cmd += f"{r}"
  if n < top_n-1:
    pymol_cmd += " "

print("VMD selection command:")
print(pymol_cmd)

