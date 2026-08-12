from scipy.io import savemat
from pathlib import Path
import torch
import numpy as np

def save_output(output):
    save_path = Path("simulation_output.mat")
    savemat(save_path, {"output": output["neurons"]["Dynamic"]["ron"]["spikes_holder"].detach().cpu().numpy().astype(np.uint8),"losses": output["neurons"]["BookKeeping"],'params': {'strf_gain': output["neurons"]["Learnable"]["STRF_gain_tracker"].detach().cpu().numpy(), 'strf_alpha': output["neurons"]["Learnable"]["STRF_alpha_tracker"].detach().cpu().numpy(),'output_ad': output["neurons"]["Learnable"]["ron"]["output_ad_tracker"].detach().cpu().numpy(),'abs_ref': output["neurons"]["Learnable"]["ron"]["abs_ref_tracker"].detach().cpu().numpy(),'rel_ref_a': output["neurons"]["Learnable"]["ron"]["rel_ref_a_tracker"].detach().cpu().numpy(),'rel_ref_b': output["neurons"]["Learnable"]["ron"]["rel_ref_b_tracker"].detach().cpu().numpy(),'rel_ref_c': output["neurons"]["Learnable"]["ron"]["rel_ref_c_tracker"].detach().cpu().numpy(),'on_ron_gsyn': output["synapses"]["Learnable"]["on_ron"]["gSYN_tracker"].detach().cpu().numpy(),'off_ron_gsyn': output["synapses"]["Learnable"]["off_ron"]["gSYN_tracker"].detach().cpu().numpy(),'sonoff_ron_gsyn': output["synapses"]["Learnable"]["sonoff_ron"]["gSYN_tracker"].detach().cpu().numpy(),'on_sonoff_gsyn': output["synapses"]["Learnable"]["on_sonoff"]["gSYN_tracker"].detach().cpu().numpy(),'off_sonoff_gsyn': output["synapses"]["Learnable"]["off_sonoff"]["gSYN_tracker"].detach().cpu().numpy()}}, do_compression=True)
