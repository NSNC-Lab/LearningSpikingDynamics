from scipy.io import savemat
from pathlib import Path
import torch
import numpy as np

def save_output(output):
    save_path = Path("simulation_output.mat")
    savemat(save_path, {"output": output["neurons"]["Dynamic"]["ron"]["spikes_holder"].detach().cpu().numpy().astype(np.uint8),"losses": output["neurons"]["BookKeeping"],'params': {'strf_gain': output["neurons"]["Learnable"]["STRF_gain"].detach().cpu().numpy(), 'strf_alpha': output["neurons"]["Learnable"]["STRF_alpha"].detach().cpu().numpy(),'output_ad': output["neurons"]["Learnable"]["ron"]["output_ad"].detach().cpu().numpy(),'abs_ref': output["neurons"]["Learnable"]["ron"]["abs_ref"].detach().cpu().numpy(),'rel_ref_a': output["neurons"]["Learnable"]["ron"]["rel_ref_a"].detach().cpu().numpy(),'rel_ref_b': output["neurons"]["Learnable"]["ron"]["rel_ref_b"].detach().cpu().numpy(),'rel_ref_c': output["neurons"]["Learnable"]["ron"]["rel_ref_c"].detach().cpu().numpy(),'on_ron_gsyn': output["synapses"]["Learnable"]["on_ron"]["gSYN"].detach().cpu().numpy(),'off_ron_gsyn': output["synapses"]["Learnable"]["off_ron"]["gSYN"].detach().cpu().numpy(),'sonoff_ron_gsyn': output["synapses"]["Learnable"]["sonoff_ron"]["gSYN"].detach().cpu().numpy(),'on_sonoff_gsyn': output["synapses"]["Learnable"]["on_sonoff"]["gSYN"].detach().cpu().numpy(),'off_sonoff_gsyn': output["synapses"]["Learnable"]["off_sonoff"]["gSYN"].detach().cpu().numpy()}}, do_compression=True)
