#! /home/niklashalbwedl/bin_custom/micromamba/envs/design/bin/python3 -u

import os
import re

from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.shared.utils import copy_dict
from colabdesign.af.alphafold.common import residue_constants
import jax
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

from functions import *


#Directory with Alphafold weights
params_dir="/home/niklashalbwedl/henmount/apps"

#PDB file of target
pdb = "/home/niklashalbwedl/henmount/master/07_8JJS/02_af_design/A_cleaned.pdb"

#set target chains and hotspot residues (from pocket search)
target_chain = "A"
target_hotspot = "A7,A9,A56,A58,A61,A62,A64,A68,A69,A71,A72,A78,A92,A95,A96,A98,A99,A102,A103"
if target_hotspot == "": target_hotspot = None
target_flexible = False

#set binder length and initial AA sequence. set binder_seq = "" for random init. seq.
binder_len = 11
binder_seq = "RHKPGTFEQLC"
if len(binder_seq) > 0:
  binder_len = len(binder_seq)
else:
  binder_seq = None

#offset to get cyclic and not linear peptide
cyclic_offset = True

#bugfixed version of cyclic offset (True)
bugfix = True

#initialize PeptideLoss (from functions)
pep_loss = PeptideLoss(n_pep_res=binder_len, hotspot_res=target_hotspot, bound=100)

#AF2 parameters (MCMC opt. uses only 1 model)
use_multimer = True
num_recycles = 0 #@param ["0", "1", "3", "6"] {type:"raw"}
num_models = 5 #@param ["1", "2", "3", "4", "5"]

#model parameters
x = {"pdb_filename":pdb,
     "chain":target_chain,
     "binder_len":binder_len,
     "hotspot":target_hotspot,
     "use_multimer":use_multimer,
     "rm_target_seq":target_flexible}

#set model parameters and losses
if "x_prev" not in dir() or x != x_prev:
  clear_mem()
  model = mk_afdesign_model(
    protocol="binder",
    use_multimer=x["use_multimer"],
    num_recycles=num_recycles,
    recycle_mode="sample", loss_callback=[pep_loss.cis_loss,pep_loss.com_loss], data_dir=params_dir)
  model.prep_inputs(**x, ignore_missing=False)
  model.opt["weights"]["cis"] = 1.0
  model.opt["weights"]["com_loss"] = 1.0
  print("weights", model.opt["weights"])
  x_prev = copy_dict(x)
  print("target length:", model._target_len)
  print("binder length:", model._binder_len)
  binder_len = model._binder_len

# Add cyclic offset if set True
if cyclic_offset:
  if bugfix:
      print("Set bug fixed cyclic peptide complex offset. The cyclic peptide binder will be hallucinated.")
      add_cyclic_offset(model, bug_fix=True)
  else:
      print("Set not bug fixed cyclic peptide complex offset. The cyclic peptide binder will be hallucinated.")
      add_cyclic_offset(model, bug_fix=False)
else:
  print("Don't set cyclic offset. The linear peptide binder will be hallucionated.")


# Set optimizer for sequence optimization

optimizer = "mcmc" #@param ["pssm_semigreedy", "3stage", "semigreedy", "pssm", "logits", "soft", "hard"]
#`pssm_semigreedy` - uses the designed PSSM to bias semigreedy opt. (Recommended by authors)
#`my_pssm_semigreedy` - uses custom semigreedy
#`3stage` - gradient based optimization (GD) (logits → soft → hard)
#`semigreedy` - tries X random mutations, accepts those that decrease loss
#`my_semigreedy` - decrease mutation rate after mutation with low loss
#`hard` - GD optimize one_hot(logits) inputs (discrete)
#`mcmc` - MCMC simulated annealing

##advanced GD settings (only for 3stage, hard and pssm_semigreedy)
GD_method = "adam" #@param ["adabelief", "adafactor", "adagrad", "adam", "adamw", "fromage", "lamb", "lars", "noisy_sgd", "dpsgd", "radam", "rmsprop", "sgd", "sm3", "yogi"]
learning_rate = 0.01 #@param {type:"raw"}
norm_seq_grad = True
dropout = True

model.restart(seq=binder_seq)
model.set_optimizer(optimizer=GD_method,
                    learning_rate=learning_rate,
                    norm_seq_grad=norm_seq_grad)
models = model._model_names[:num_models]

flags = {"num_recycles":num_recycles,
         "models":models,
         "dropout":dropout}

if optimizer == "3stage":
  model.design_3stage(120, 60, 10, **flags)
  pssm = softmax(model._tmp["seq_logits"],-1)

if optimizer == "my_pssm_semigreedy":
  model.my_design_pssm_semigreedy(120, 128, **flags)
  pssm = softmax(model._tmp["seq_logits"],1)

if optimizer == "my_semigreedy":
  model.my_design_pssm_semigreedy(0, 256, **flags)
  pssm = None

if optimizer == "pssm_semigreedy":
  model.design_pssm_semigreedy(120, 128, **flags)
  pssm = softmax(model._tmp["seq_logits"],1)

if optimizer == "semigreedy":
  model.design_pssm_semigreedy(0, 256, **flags)
  pssm = None

if optimizer == "mcmc":
  model._design_mcmc(steps=1000, mutation_rate=2, T_init=0.01,half_life=200, **flags)

if optimizer == "pssm":
  model.design_logits(120, e_soft=1.0, num_models=1, ramp_recycles=True, **flags)
  model.design_soft(32, num_models=1, **flags)
  flags.update({"dropout":False,"save_best":True})
  model.design_soft(10, num_models=num_models, **flags)
  pssm = softmax(model.aux["seq"]["logits"],-1)

model.save_pdb(f"{model.protocol}.pdb")

