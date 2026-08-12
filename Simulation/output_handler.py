from pathlib import Path

import numpy as np
from scipy.io import savemat


def to_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value,"detach") else np.asarray(value)


def tracked_params(output,epochs_completed):
    neurons, synapses = output["neurons"]["Learnable"], output["synapses"]["Learnable"]
    ron = neurons["ron"]
    values = {"strf_gain":neurons["STRF_gain_tracker"], "strf_alpha":neurons["STRF_alpha_tracker"],
              "output_ad":ron["output_ad_tracker"], "abs_ref":ron["abs_ref_tracker"],
              "rel_ref_a":ron["rel_ref_a_tracker"], "rel_ref_b":ron["rel_ref_b_tracker"],
              "rel_ref_c":ron["rel_ref_c_tracker"],
              **{f"{name}_gsyn":synapses[name]["gSYN_tracker"] for name in
                 ("on_ron","off_ron","sonoff_ron","on_sonoff","off_sonoff")}}
    return {name:to_numpy(value[...,:epochs_completed]) for name,value in values.items()}


def checkpoint_params(output):
    neurons, synapses = output["neurons"]["Learnable"], output["synapses"]["Learnable"]
    ron = neurons["ron"]
    values = {"strf_gain":neurons["STRF_gain"], "strf_alpha":neurons["STRF_alpha"],
              "output_ad":ron["output_ad"], "abs_ref":ron["abs_ref"],
              "rel_ref_a":ron["rel_ref_a"], "rel_ref_b":ron["rel_ref_b"],
              "rel_ref_c":ron["rel_ref_c"],
              **{f"{name}_gsyn":synapses[name]["gSYN"] for name in
                 ("on_ron","off_ron","sonoff_ron","on_sonoff","off_sonoff")}}
    return {name:to_numpy(value) for name,value in values.items()}


def scalar_history(values):
    return np.asarray([float(value.detach().cpu()) if hasattr(value,"detach") else float(value)
                       for value in values],dtype=np.float32)


def save_output(output,path="simulation_output.mat",epochs_completed=None):
    neurons, ron = output["neurons"], output["neurons"]["Dynamic"]["ron"]
    epochs_completed = len(neurons["BookKeeping"]) if epochs_completed is None else epochs_completed
    adam = neurons["Adam"]
    data = {"output":to_numpy(ron["spikes_holder"]).astype(np.uint8),
            "losses":scalar_history(neurons["BookKeeping"]),
            "cv_losses":scalar_history(neurons.get("CVBookKeeping",[])),
            "sse_losses_all":to_numpy(ron["all_sse_loss"][...,:epochs_completed]),
            "cv_losses_all":to_numpy(ron["all_CV_loss"][...,:epochs_completed]),
            "params":tracked_params(output,epochs_completed),
            "checkpoint_params":checkpoint_params(output),
            "adam":{"m":to_numpy(adam["m"]),"v":to_numpy(adam["v"]),
                    "t":np.asarray([[adam["t"]]],dtype=np.int64)},
            "epochs_completed":np.asarray([[epochs_completed]],dtype=np.int64)}
    savemat(Path(path),data,do_compression=True)
