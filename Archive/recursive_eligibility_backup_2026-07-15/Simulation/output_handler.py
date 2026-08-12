from pathlib import Path

import numpy as np
from scipy.io import savemat


def to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def tracked_params(output):
    return {
        "strf_gain": to_numpy(output["neurons"]["Learnable"]["STRF_gain_tracker"]),
        "strf_alpha": to_numpy(output["neurons"]["Learnable"]["STRF_alpha_tracker"]),
        "output_ad": to_numpy(output["neurons"]["Learnable"]["ron"]["output_ad_tracker"]),
        "abs_ref": to_numpy(output["neurons"]["Learnable"]["ron"]["abs_ref_tracker"]),
        "rel_ref_a": to_numpy(output["neurons"]["Learnable"]["ron"]["rel_ref_a_tracker"]),
        "rel_ref_b": to_numpy(output["neurons"]["Learnable"]["ron"]["rel_ref_b_tracker"]),
        "rel_ref_c": to_numpy(output["neurons"]["Learnable"]["ron"]["rel_ref_c_tracker"]),
        "on_ron_gsyn": to_numpy(output["synapses"]["Learnable"]["on_ron"]["gSYN_tracker"]),
        "off_ron_gsyn": to_numpy(output["synapses"]["Learnable"]["off_ron"]["gSYN_tracker"]),
        "sonoff_ron_gsyn": to_numpy(output["synapses"]["Learnable"]["sonoff_ron"]["gSYN_tracker"]),
        "on_sonoff_gsyn": to_numpy(output["synapses"]["Learnable"]["on_sonoff"]["gSYN_tracker"]),
        "off_sonoff_gsyn": to_numpy(output["synapses"]["Learnable"]["off_sonoff"]["gSYN_tracker"]),
    }


def checkpoint_params(output):
    return {
        "strf_gain": to_numpy(output["neurons"]["Learnable"]["STRF_gain"]),
        "strf_alpha": to_numpy(output["neurons"]["Learnable"]["STRF_alpha"]),
        "output_ad": to_numpy(output["neurons"]["Learnable"]["ron"]["output_ad"]),
        "abs_ref": to_numpy(output["neurons"]["Learnable"]["ron"]["abs_ref"]),
        "rel_ref_a": to_numpy(output["neurons"]["Learnable"]["ron"]["rel_ref_a"]),
        "rel_ref_b": to_numpy(output["neurons"]["Learnable"]["ron"]["rel_ref_b"]),
        "rel_ref_c": to_numpy(output["neurons"]["Learnable"]["ron"]["rel_ref_c"]),
        "on_ron_gsyn": to_numpy(output["synapses"]["Learnable"]["on_ron"]["gSYN"]),
        "off_ron_gsyn": to_numpy(output["synapses"]["Learnable"]["off_ron"]["gSYN"]),
        "sonoff_ron_gsyn": to_numpy(output["synapses"]["Learnable"]["sonoff_ron"]["gSYN"]),
        "on_sonoff_gsyn": to_numpy(output["synapses"]["Learnable"]["on_sonoff"]["gSYN"]),
        "off_sonoff_gsyn": to_numpy(output["synapses"]["Learnable"]["off_sonoff"]["gSYN"]),
    }


def adam_state(output):
    adam = output["neurons"]["Adam"]
    return {
        "m": to_numpy(adam["m"]),
        "v": to_numpy(adam["v"]),
        "t": np.array([[adam["t"]]], dtype=np.int64),
    }


def save_output(output):
    save_path = Path("simulation_output.mat")
    savemat(
        save_path,
        {
            "output": to_numpy(output["neurons"]["Dynamic"]["ron"]["spikes_holder"]).astype(np.uint8),
            "losses": output["neurons"]["BookKeeping"],
            "firing_rates_hz": np.asarray(output["neurons"].get("FiringRateBookKeeping", []), dtype=np.float32),
            "cv_losses": np.asarray(output["neurons"].get("CVBookKeeping", []), dtype=np.float32),
            "params": tracked_params(output),
            "checkpoint_params": checkpoint_params(output),
            "adam": adam_state(output),
        },
        do_compression=True,
    )
