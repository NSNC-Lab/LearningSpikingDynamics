from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch 

def load_gt_data(args):
    repo_root = Path(__file__).resolve().parents[1]

    mat = loadmat(repo_root / args["paths"]["data"],variable_names=["all_data"],squeeze_me=True,struct_as_record=False)
    all_data = mat["all_data"]
    bin_edges = np.arange(0, (args["simulation"]["sim_len"])/10000, args["simulation"]["PSTH_granularity"]/10000)
    psth_holder = torch.zeros((len(args["simulation"]["cell_targets"]), int(np.round(args["simulation"]["sim_len"]/args["simulation"]["PSTH_granularity"]))), dtype=torch.float32, device=torch.device(args["simulation"]["device"]))
    raster_holder = torch.zeros((len(args["simulation"]["cell_targets"]), 10, int(np.round(args["simulation"]["sim_len"]))), dtype=torch.float32, device=torch.device(args["simulation"]["device"]))

    for k,cell in enumerate(args["simulation"]["cell_targets"]):
        timestamps = all_data[cell-1].ctrl_tar1_timestamps
        if args["simulation"]["data_target"] == "peak":
            if all_data[cell-1].tuning_type == 'contra-tuned':
                angle = 0
            elif all_data[cell-1].tuning_type == '45°-tuned':
                angle = 1
            elif all_data[cell-1].tuning_type == 'center-tuned':
                angle = 2
            else:
                angle = 3
            
            timestamps_peak = timestamps[:,angle]
            
            for m in range(10):
                timestamps_peak[m] = np.asarray(timestamps_peak[m])
                if timestamps_peak[m].size > 0: #Just makes sure there are acutally spikes or else it will fail out
                    cut_timesteps = timestamps_peak[m][(timestamps_peak[m] >= 0) & (timestamps_peak[m] <= 2.9801)]
                    counts, bin_edges = np.histogram(cut_timesteps, bins=bin_edges) 
                    psth_holder[k,:] += torch.tensor(counts, dtype=torch.float32, device=torch.device(args["simulation"]["device"]))
                    raster_holder[k,m,(cut_timesteps*10000).astype(np.int64)] = 1

    return {'psth_holder' : psth_holder, 'raster_holder' : raster_holder}



    