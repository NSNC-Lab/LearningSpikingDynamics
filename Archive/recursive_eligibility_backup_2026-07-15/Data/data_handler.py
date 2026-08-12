from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch 

def load_gt_data(args):
    repo_root = Path(__file__).resolve().parents[1]

    mat = loadmat(repo_root / args["paths"]["data"],variable_names=["all_data"],squeeze_me=True,struct_as_record=False)
    all_data = mat["all_data"]
    if args["simulation"]["data_target"] != "peak":
        raise NotImplementedError("data_target currently supports only 'peak'; returning zero targets would collapse training.")

    dt_seconds = args["simulation"]["dt"] / 1000.0
    complete_bins = int(np.floor(args["simulation"]["sim_len"]/args["simulation"]["PSTH_granularity"]))
    bin_edges = np.arange(complete_bins + 1) * args["simulation"]["PSTH_granularity"] * dt_seconds
    max_binned_time = bin_edges[-1]
    psth_holder = torch.zeros((len(args["simulation"]["cell_targets"]), complete_bins), dtype=torch.float32, device=torch.device(args["simulation"]["device"]))
    raster_holder = torch.zeros((len(args["simulation"]["cell_targets"]), 10, int(np.round(args["simulation"]["sim_len"]))), dtype=torch.float32, device=torch.device(args["simulation"]["device"]))

    for k,cell in enumerate(args["simulation"]["cell_targets"]):
        timestamps = all_data[cell-1].ctrl_tar1_timestamps
        if all_data[cell-1].tuning_type == 'contra-tuned':
            angle = 0
        elif '45' in str(all_data[cell-1].tuning_type):
            angle = 1
        elif all_data[cell-1].tuning_type == 'center-tuned':
            angle = 2
        else:
            angle = 3

        timestamps_peak = timestamps[:,angle]

        for m in range(10):
            timestamps_peak[m] = np.asarray(timestamps_peak[m])
            if timestamps_peak[m].size > 0: #Just makes sure there are acutally spikes or else it will fail out
                cut_timesteps = timestamps_peak[m][(timestamps_peak[m] >= 0) & (timestamps_peak[m] < max_binned_time)]
                counts, bin_edges = np.histogram(cut_timesteps, bins=bin_edges)
                psth_holder[k,:] += torch.tensor(counts, dtype=torch.float32, device=torch.device(args["simulation"]["device"]))
                raster_holder[k,m,np.floor(cut_timesteps/dt_seconds).astype(np.int64)] = 1

    return {'psth_holder' : psth_holder, 'raster_holder' : raster_holder}
