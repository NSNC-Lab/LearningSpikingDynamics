from pathlib import Path

from scipy.io import wavfile
from scipy.signal import ShortTimeFFT, windows, fftconvolve

import numpy as np
import math
import torch

import matplotlib.pyplot as plt

#Note! When we add maskers to this we will need a way to do the 10 maskers and 2 targets automatically so that
# we are not storing 12 varibles per operation that we perform.

def create_input_fr(args,states,device):

    #Load in the stimuli
    target1 = load_target(args,1)

    #Compute FFT of targets
    target1_fft = compute_stft(target1, args)

    #Create STRF (Frequency from target1_fft is fed in to match the range of frequencies use in the FFTs)
    strf = create_strf(args,states,target1_fft['f'])

    #Convolve STRF with target FFT to get firing rates
    rates = compute_convolution(args, strf, target1_fft)

    #Compute onset and offset rates and add gain
    rates_on_off = compute_rates(args, states, rates,device)

    return rates_on_off

def load_target(args, target_num):
    REPO_ROOT = Path(__file__).resolve().parents[1] 
    fs, data = wavfile.read(REPO_ROOT/args['paths']['stimuli'] / f"200k_target{target_num}.wav")
    return {'fs': fs, 'data': data}

def compute_stft(data, args):

    # This creates a time window in order to enfoce the frequency resolution given by fband.
    winLength = int(int(args['strf']['fft_parms']['nstd']/(args['strf']['fft_parms']['fband']*2.0*math.pi)*data['fs'])/2)*2    # Window length in number of points

    #Take the STFT. win - gaussian smooths the frame you take the STFT of, and hop makes it so that dt matches the simulation. p0 and p1 make it so that there is no auto-padding from SFT
    win = windows.gaussian(winLength, std=winLength / args['strf']['fft_parms']['nstd'])
    SFT = ShortTimeFFT(win, hop=int(data['fs']*args['simulation']['dt']/1000), fs=data['fs'], fft_mode="onesided")
    S = SFT.stft(data['data'],p0=0,p1=int(np.floor(len(data['data']) / SFT.hop) + 1))

    #Cut down the stimuli to be between the high and low frequencies specified.
    maxIndx = np.where(SFT.f >= args['strf']['fft_parms']['high_freq'])[0][0]
    minIndx = np.where(SFT.f < args['strf']['fft_parms']['low_freq'])[0][-1]+1
    normedS = np.abs(S[minIndx:maxIndx+1, :])

    #Convert to dB and add noise floor. In this form we can convolve with the STRF.
    spec = 20 * np.log10(np.maximum(normedS, 1e-12) / np.max(normedS)) + args["strf"]["fft_parms"]["dbnoise"]
    spec[spec < 0] = 0

    return {'spec': spec, 't': SFT.t(len(data['data']), p0=0, p1=int(np.floor(len(data['data']) / SFT.hop) + 1)), 'f': SFT.f[minIndx:maxIndx+1]}

def create_strf(args,states,f):
    
    strf = {}
    strf_deriv = {}
    strf_deriv_alpha = {}
    strf_deriv_gain = {}

    #Temporal kernel -- Derivative is also calcualted here.
    kernel_steps = int(args['strf']['strf_params']['maxdelay'])
    strf['t'] = torch.arange(kernel_steps, device=torch.device(args['simulation']['device']), dtype=torch.float32) * args['simulation']['dt']
    strf['H'] = torch.exp(-strf['t'][:,None,None]/states['neurons']['Learnable']['STRF_alpha'][None,:,:])*(args['strf']['strf_params']['SC1']*(strf['t'][:,None,None]/states['neurons']['Learnable']['STRF_alpha'][None,:,:])**args['strf']['strf_params']['N1']/math.factorial(args['strf']['strf_params']['N1']) - args['strf']['strf_params']['SC2'] * (strf['t'][:,None,None]/states['neurons']['Learnable']['STRF_alpha'][None,:,:])**args['strf']['strf_params']['N2']/math.factorial(args['strf']['strf_params']['N2']))
    strf_deriv['H_epsilon'] = torch.exp(-strf['t'][:,None,None]/states['neurons']['Learnable']['STRF_alpha'][None,:,:])
    strf_deriv['H_gamma'] = (args['strf']['strf_params']['SC1']*(strf['t'][:,None,None]/states['neurons']['Learnable']['STRF_alpha'][None,:,:])**args['strf']['strf_params']['N1']/math.factorial(args['strf']['strf_params']['N1']) - args['strf']['strf_params']['SC2'] * (strf['t'][:,None,None]/states['neurons']['Learnable']['STRF_alpha'][None,:,:])**args['strf']['strf_params']['N2']/math.factorial(args['strf']['strf_params']['N2']))
    strf_deriv['H'] = strf_deriv['H_epsilon']*((args['strf']['strf_params']['SC1']*args['strf']['strf_params']['N1']*(-strf['t'][:,None,None]/states['neurons']['Learnable']['STRF_alpha'][None,:,:]**2))*(strf['t'][:,None,None]/states['neurons']['Learnable']['STRF_alpha'][None,:,:])**(args['strf']['strf_params']['N1']-1)/math.factorial(args['strf']['strf_params']['N1']) - (args['strf']['strf_params']['SC2']*args['strf']['strf_params']['N2']*(-strf['t'][:,None,None]/states['neurons']['Learnable']['STRF_alpha'][None,:,:]**2)) * (strf['t'][:,None,None]/states['neurons']['Learnable']['STRF_alpha'][None,:,:])**(args['strf']['strf_params']['N2']-1)/math.factorial(args['strf']['strf_params']['N2'])) + strf_deriv['H_epsilon']*strf_deriv['H_gamma']*(strf['t'][:,None,None]/states['neurons']['Learnable']['STRF_alpha'][None,:,:]**2)
                
    #Frequency kernel
    strf['G'] = torch.exp(-0.5*((torch.tensor(f,device=torch.device(args['simulation']['device']))-args['strf']['strf_params']['f0'])/args['strf']['strf_params']['BW'])**2)* torch.cos(2*torch.pi*args['strf']['strf_params']['BSM']*(torch.tensor(f,device=torch.device(args['simulation']['device']))-args['strf']['strf_params']['f0']))


    #Add gain.
    strf_deriv_alpha['H'] = strf_deriv['H'] * states['neurons']['Learnable']['STRF_gain'][None,:,:]
    strf_deriv_gain['H'] = strf['H']
    strf['H'] = strf['H'] * states['neurons']['Learnable']['STRF_gain'][None,:,:]


    #Outer Product to get STRF
    strf['w1']=strf['G'][:,None,None,None]*strf['H'][None,:,:,:]
    strf_deriv['w1'] = strf['G'][:,None,None,None]*strf_deriv['H'][None,:,:,:]

    return {'strf': strf, 'strf_deriv_alpha': strf_deriv_alpha, 'strf_deriv_gain': strf_deriv_gain}

def compute_convolution(args, strf, target_fft):

    rate_params = args["strf"].get("rate_params", {})
    stim_gain = rate_params.get("stimGain", 0.5)
    mean_rate = rate_params.get("mean_rate", 0.1)

    drive = stim_gain * np.einsum("ft,f->t", target_fft["spec"], strf["strf"]["G"].detach().cpu().numpy())
    a = fftconvolve(drive[:, None, None], strf["strf"]["H"].detach().cpu().numpy(), mode="full", axes=0)
    a = mean_rate * a[:len(target_fft["t"])]
    crop_start = int(args['strf']['strf_params']['maxdelay'])
    a = a[crop_start:]

    a_deriv_alpha = fftconvolve(drive[:, None, None], strf["strf_deriv_alpha"]["H"].detach().cpu().numpy(), mode="full", axes=0)
    a_deriv_gain = fftconvolve(drive[:, None, None], strf["strf_deriv_gain"]["H"].detach().cpu().numpy(), mode="full", axes=0)
    a_deriv_alpha = mean_rate * a_deriv_alpha[:len(target_fft["t"])]
    a_deriv_gain = mean_rate * a_deriv_gain[:len(target_fft["t"])]
    a_deriv_alpha = a_deriv_alpha[crop_start:]
    a_deriv_gain = a_deriv_gain[crop_start:]

    return {'a': a, 'a_deriv_alpha': a_deriv_alpha, 'a_deriv_gain': a_deriv_gain}


def causal_running_max_offset(values, alpha_derivative, gain_derivative):
    """Create a causal falling-drive channel and its piecewise derivatives.

    ``offset[t] = running_max(values[:t+1]) - values[t]``.  Unlike a global
    maximum, this cannot use a future response to create spikes at time zero.
    At max-switching ties the earlier maximum is retained, defining a stable
    one-sided subgradient.
    """
    offset = np.zeros_like(values)
    offset_alpha_derivative = np.zeros_like(alpha_derivative)
    offset_gain_derivative = np.zeros_like(gain_derivative)

    running_value = values[0].copy()
    running_alpha_derivative = alpha_derivative[0].copy()
    running_gain_derivative = gain_derivative[0].copy()
    for timestep in range(1, values.shape[0]):
        new_maximum = values[timestep] > running_value
        running_value = np.where(new_maximum, values[timestep], running_value)
        running_alpha_derivative = np.where(
            new_maximum, alpha_derivative[timestep], running_alpha_derivative
        )
        running_gain_derivative = np.where(
            new_maximum, gain_derivative[timestep], running_gain_derivative
        )

        raw_offset = running_value - values[timestep]
        active = raw_offset > 0
        offset[timestep] = np.where(active, raw_offset, 0.0)
        offset_alpha_derivative[timestep] = np.where(
            active,
            running_alpha_derivative - alpha_derivative[timestep],
            0.0,
        )
        offset_gain_derivative[timestep] = np.where(
            active,
            running_gain_derivative - gain_derivative[timestep],
            0.0,
        )

    return offset, offset_alpha_derivative, offset_gain_derivative

def compute_rates(args, states, rates, device):

    # Match the old onset implementation: gain is already baked into H, so the
    # rate is the rectified, scaled convolution output.
    strf_gain = states['neurons']['Learnable']['STRF_gain'][None,:,:].detach().cpu().numpy()


    #ungained_rate = rates['a'] / np.maximum(strf_gain, 1e-12)
    #centered_rate = rates['a'] - np.mean(rates['a'], axis=0, keepdims=True)
    #centered_rate_deriv = rates['a_deriv'] - np.mean(rates['a_deriv'], axis=0, keepdims=True)
    #centered_ungained_rate = ungained_rate - np.mean(ungained_rate, axis=0, keepdims=True)

    #onset_rate_gain_deriv = np.maximum(ungained_rate, 0)
    #offset_rate_gain_deriv = np.maximum(-centered_ungained_rate, 0)
    onset_rate = np.maximum(rates['a'], 0)

    onset_rate_gain_deriv = rates['a_deriv_gain'] * (rates['a'] > 0)
    onset_rate_deriv = rates['a_deriv_alpha'] * (rates['a'] > 0)

    offset_mode = args.get("preprocessing", {}).get(
        "offset_mode", "causal_running_max"
    )
    if offset_mode == "causal_running_max":
        offset_rate, offset_rate_deriv, offset_rate_gain_deriv = causal_running_max_offset(
            rates['a'], rates['a_deriv_alpha'], rates['a_deriv_gain']
        )
    elif offset_mode == "global_max":
        offset_drive = -rates['a'] + np.max(rates['a'], axis=0, keepdims=True)
        offset_active = offset_drive > 0
        offset_rate = np.maximum(offset_drive, 0)
        max_index = np.argmax(rates['a'], axis=0, keepdims=True)
        offset_rate_gain_deriv = (
            -rates['a_deriv_gain']
            + np.take_along_axis(rates['a_deriv_gain'], max_index, axis=0)
        ) * offset_active
        offset_rate_deriv = (
            -rates['a_deriv_alpha']
            + np.take_along_axis(rates['a_deriv_alpha'], max_index, axis=0)
        ) * offset_active
    else:
        raise ValueError(
            f"Unknown preprocessing.offset_mode={offset_mode!r}; "
            "expected 'causal_running_max' or 'global_max'."
        )

    return {'onset_rate': torch.tensor(onset_rate, device=device),
            'offset_rate': torch.tensor(offset_rate, device=device),
            'onset_rate_deriv': torch.tensor(onset_rate_deriv, device=device),
            'offset_rate_deriv': torch.tensor(offset_rate_deriv, device=device),
            'onset_rate_gain_deriv': torch.tensor(onset_rate_gain_deriv, device=device),
            'offset_rate_gain_deriv': torch.tensor(offset_rate_gain_deriv, device=device)}



