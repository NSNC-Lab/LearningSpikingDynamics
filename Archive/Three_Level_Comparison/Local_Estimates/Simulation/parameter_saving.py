"""Reusable parameter history and optional checkpoint saving.

Call initialize once, record immediately before Adam, and save at the end.
Use save(..., checkpoint=True) to also store post-update parameters and Adam.
"""

from pathlib import Path
import numpy as np
import torch
from scipy.io import savemat, loadmat

SYNAPSES = ("on_ron","off_ron","sonoff_ron","on_sonoff","off_sonoff")

def live(states):
    neurons, synapses = states["neurons"]["Learnable"], states["synapses"]["Learnable"]
    ron = neurons["ron"]
    return {"strf_gain":neurons["STRF_gain"], "strf_alpha":neurons["STRF_alpha"],
            "output_ad":ron["output_ad"], "abs_ref":ron["abs_ref"],
            "rel_ref_a":ron["rel_ref_a"], "rel_ref_b":ron["rel_ref_b"], "rel_ref_c":ron["rel_ref_c"],
            **{f"{name}_gsyn":synapses[name]["gSYN"] for name in SYNAPSES}}

def array(value):
    return value.detach().cpu().numpy() if hasattr(value,"detach") else np.asarray(value)

def initialize(states,epochs):
    states["parameter_history"] = {name:np.zeros((*value.shape,epochs),np.float32) for name,value in live(states).items()}
    states["epochs_completed"] = 0

def record(states,epoch):
    for name,value in live(states).items(): 
        states["parameter_history"][name][...,epoch] = array(value)
        states["epochs_completed"] = epoch+1

def save(states,path="simulation_output.mat",checkpoint=False):
    neurons, ron = states["neurons"], states["neurons"]["Dynamic"]["ron"]
    data = {"output":array(ron["spikes_holder"]).astype(np.uint8), "losses":np.asarray([float(x) for x in neurons["BookKeeping"]],np.float32),
            "params":{name:value[...,:states["epochs_completed"]] for name,value in states["parameter_history"].items()}}
    data.update({mat_name:array(ron[state_name]) for state_name,mat_name in (("all_CV_loss","cv_losses_all"),("all_sse_loss","sse_losses_all")) if state_name in ron})
    if "CVBookKeeping" in neurons: data["cv_losses"] = np.asarray([float(x) for x in neurons["CVBookKeeping"]],np.float32)
    if checkpoint:
        adam = neurons["Adam"]
        data.update({"checkpoint_params":{name:array(value) for name,value in live(states).items()},
                     "adam":{"m":array(adam["m"]),"v":array(adam["v"]),"t":adam["t"]}, "epochs_completed":states["epochs_completed"]})
    savemat(Path(path),data,do_compression=True)

def restore(states,path):
    saved = loadmat(path,variable_names=("checkpoint_params","adam"),simplify_cells=True)
    with torch.no_grad():
        for name,target in live(states).items():
            value = torch.as_tensor(saved["checkpoint_params"][name],
                                    device=target.device,dtype=target.dtype)
            target.copy_(value.reshape_as(target))

        adam = states["neurons"]["Adam"]
        for name in ("m","v"):
            value = torch.as_tensor(saved["adam"][name],
                                    device=adam[name].device,dtype=adam[name].dtype)
            adam[name].copy_(value.reshape_as(adam[name]))
        adam["t"] = int(saved["adam"]["t"])
        print(f"Restored parameters from {path}")