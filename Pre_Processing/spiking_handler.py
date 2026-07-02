import numpy as np
import torch

from scipy.io import loadmat
from pathlib import Path

def generate_spike_trains(rates, args, device):

    onset_offset_spks = generate_model_inputs(rates,args,device)
    noise_spks = generate_noise_activity(rates,args,device)

    return {'onset_offset_spks': onset_offset_spks, 'noise_spks': noise_spks}

def generate_model_inputs(rates,args,device):

    #P(spiking) is approximately rate*dt (/1000 to convert ms to s). I found 1* converts from bool to int.
    onset_spks = 1 * ((rates['onset_rate']*args['simulation']['dt']/1000)[:,:,:,None] > torch.rand([rates['onset_rate'].shape[0],rates['onset_rate'].shape[1],rates['onset_rate'].shape[2],10], device=device))
    onset_spks = onset_spks.permute(1,3,2,0)
    offset_spks = 1 * ((rates['offset_rate']*args['simulation']['dt']/1000)[:,:,:,None] > torch.rand([rates['offset_rate'].shape[0],rates['offset_rate'].shape[1],rates['offset_rate'].shape[2],10], device=device))
    offset_spks = offset_spks.permute(1,3,2,0)

    return {'onset_spks': onset_spks, 'offset_spks': offset_spks}

def generate_noise_activity(rates,args,device):

    #Load in the data.
    REPO_ROOT = Path(__file__).resolve().parents[1] 
    m = np.load(REPO_ROOT/args['paths']['spontaneous_activity'])

    #The 10 here is the # of trials. This is always 10 in the data.
    noise_spks = np.zeros((10, len(args['simulation']['cell_targets']),rates['offset_rate'].shape[0]))

    for count,k in enumerate(args['simulation']['cell_targets']):
        noise_spks[:,count,:] = 1 * ((np.mean(m[k-1,0:4])*args['simulation']['dt']/1000) > np.random.random((10,noise_spks.shape[2])))

    return torch.tensor(noise_spks, dtype=torch.float32, device=device)