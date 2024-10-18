#! /home/niklashalbwedl/bin_custom/micromamba/envs/design/bin/python3 -u

import json
import os
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.shared.utils import copy_dict
from functions import PeptideLoss, add_cyclic_offset

# Load parameters from the JSON configuration file
config_file = "config.json"
with open(config_file, "r") as f:
    config = json.load(f)

# Extract parameters from the JSON file
params_dir = config["params_dir"]
pdb = config["pdb"]
target_chain = config["target_chain"]
#target_hotspot = config["target_hotspot"]
target_hotspot_1 = config["target_hotspot_1"]
target_hotspot_2 = config["target_hotspot_2"]
target_flexible = config["target_flexible"]
binder_len = config["binder_len"]
binder_seq = config["binder_seq"]
cyclic_offset = config["cyclic_offset"]
rm_aa = config["rm_aa"]

target_hotspot = target_hotspot_1 + "," + target_hotspot_2

# These paramters can be kept at default
bugfix = True
use_multimer = True
num_recycles = 3
num_models = 1
optimizer = "pssm_semigreedy"
GD_method = "adam"
learning_rate = 0.01
norm_seq_grad = True
dropout = True

if target_hotspot_1 == "" or target_hotspot_2 == "":
    target_hotspot_1 = None
    target_hotspot_2 = None

if target_hotspot == ",":
    target_hotspot = None

if len(binder_seq) > 0:
    binder_len = len(binder_seq)
else:
    binder_seq = None

pep_loss = PeptideLoss(n_pep_res=binder_len, hotspot_res=target_hotspot, bound=100)

# Set up the model
x = {
    "pdb_filename": pdb,
    "chain": target_chain,
    "binder_len": binder_len,
    "hotspot": target_hotspot,
    "hotspot_1": target_hotspot_1,
    "hotspot_2": target_hotspot_2,
    "use_multimer": use_multimer,
    "rm_target_seq": target_flexible
}

# Set model parameters and losses
if "x_prev" not in dir() or x != x_prev:
    clear_mem()
    model = mk_afdesign_model(
        protocol="binder",
        use_multimer=x["use_multimer"],
        num_recycles=num_recycles,
        recycle_mode="sample",
        loss_callback=pep_loss.com_loss,
        data_dir=params_dir
    )
    model.prep_inputs(**x, rm_aa=rm_aa, ignore_missing=False)
    model.opt["weights"]["plddt"] = 1.0
    model.opt["weights"]["com_loss"] = 0.01
    #print("weights", model.opt["weights"])
    x_prev = copy_dict(x)
    print("target length:", model._target_len)
    print("binder length:", model._binder_len)
    binder_len = model._binder_len

# Add cyclic offset if set True
if cyclic_offset:
    if bugfix:
        print("Set bug-fixed cyclic peptide complex offset. The cyclic peptide binder will be hallucinated.")
        add_cyclic_offset(model, bug_fix=True)
    else:
        print("Set not bug-fixed cyclic peptide complex offset. The cyclic peptide binder will be hallucinated.")
        add_cyclic_offset(model, bug_fix=False)
else:
    print("Don't set cyclic offset. The linear peptide binder will be hallucinated.")

# Set optimizer for sequence optimization
model.restart(seq=binder_seq)
model.set_optimizer(optimizer=GD_method,
                    learning_rate=learning_rate,
                    norm_seq_grad=norm_seq_grad)
models = model._model_names[:num_models]

flags = {
    "num_recycles": num_recycles,
    "models": models,
    "dropout": dropout
}

if optimizer == "3stage":
    model.design_3stage(120, 60, 10, **flags)
    pssm = softmax(model._tmp["seq_logits"], -1)

if optimizer == "my_pssm_semigreedy":
    model.my_design_pssm_semigreedy(120, 128, **flags)
    pssm = softmax(model._tmp["seq_logits"], 1)

if optimizer == "my_semigreedy":
    model.my_design_pssm_semigreedy(0, 256, **flags)
    pssm = None

if optimizer == "pssm_semigreedy":
    model.design_pssm_semigreedy(120, 128, **flags)
    pssm = softmax(model._tmp["seq_logits"], 1)

if optimizer == "semigreedy":
    model.design_pssm_semigreedy(0, 256, **flags)
    pssm = None

if optimizer == "mcmc":
    model._design_mcmc(steps=1000, mutation_rate=1, T_init=0.01, half_life=200, **flags)

if optimizer == "pssm":
    model.design_logits(120, e_soft=1.0, num_models=1, ramp_recycles=True, **flags)
    model.design_soft(32, num_models=1, **flags)
    flags.update({"dropout": False, "save_best": True})
    model.design_soft(10, num_models=num_models, **flags)
    pssm = softmax(model.aux["seq"]["logits"], -1)

model.save_pdb(f"{model.protocol}.pdb")

