from pathlib import Path

from scipy.io import wavfile
from scipy.signal import ShortTimeFFT, windows, fftconvolve

import math
import numpy as np
import torch


def create_input_fr(args, states, device):
    target1 = load_target(args, 1)
    target1_fft = compute_stft(target1, args)
    strf = create_strf(args, states, target1_fft["f"])
    rates = compute_convolution(args, strf, target1_fft)
    return compute_rates(args, states, rates, device)


def load_target(args, target_num):
    repo_root = Path(__file__).resolve().parents[1]
    fs, data = wavfile.read(repo_root / args["paths"]["stimuli"] / f"200k_target{target_num}.wav")
    return {"fs": fs, "data": data}


def compute_stft(data, args):
    fft_args = args["strf"]["fft_parms"]
    win_length = int(int(fft_args["nstd"] / (fft_args["fband"] * 2.0 * math.pi) * data["fs"]) / 2) * 2
    hop = int(data["fs"] * args["simulation"]["dt"] / 1000)
    win = windows.gaussian(win_length, std=win_length / fft_args["nstd"])
    sft = ShortTimeFFT(win, hop=hop, fs=data["fs"], fft_mode="onesided")
    p1 = int(np.floor(len(data["data"]) / sft.hop) + 1)
    spectrum = sft.stft(data["data"], p0=0, p1=p1)

    max_index = np.where(sft.f >= fft_args["high_freq"])[0][0]
    min_index = np.where(sft.f < fft_args["low_freq"])[0][-1] + 1
    normed_spectrum = np.abs(spectrum[min_index:max_index + 1, :])

    spec = 20 * np.log10(np.maximum(normed_spectrum, 1e-12) / np.max(normed_spectrum)) + fft_args["dbnoise"]
    spec[spec < 0] = 0

    return {"spec": spec, "t": sft.t(len(data["data"]), p0=0, p1=p1), "f": sft.f[min_index:max_index + 1]}


def create_strf(args, states, f):
    params = args["strf"]["strf_params"]
    device = torch.device(args["simulation"]["device"])

    alpha = states["neurons"]["Learnable"]["STRF_alpha"].detach().to(device)
    gain = states["neurons"]["Learnable"]["STRF_gain"].detach().to(device)
    t = torch.linspace(
        0,
        params["maxdelay"] * args["simulation"]["dt"],
        int((params["maxdelay"] * args["simulation"]["dt"]) / args["simulation"]["dt"]),
        device=device,
    )
    scaled_t = t[:, None, None] / alpha[None, :, :]

    h = torch.exp(-scaled_t) * (
        params["SC1"] * scaled_t ** params["N1"] / math.factorial(params["N1"])
        - params["SC2"] * scaled_t ** params["N2"] / math.factorial(params["N2"])
    )

    h_epsilon = torch.exp(-scaled_t)
    h_gamma = (
        params["SC1"] * scaled_t ** params["N1"] / math.factorial(params["N1"])
        - params["SC2"] * scaled_t ** params["N2"] / math.factorial(params["N2"])
    )
    h_alpha_deriv = h_epsilon * (
        (params["SC1"] * params["N1"] * (-t[:, None, None] / alpha[None, :, :] ** 2))
        * scaled_t ** (params["N1"] - 1)
        / math.factorial(params["N1"])
        - (params["SC2"] * params["N2"] * (-t[:, None, None] / alpha[None, :, :] ** 2))
        * scaled_t ** (params["N2"] - 1)
        / math.factorial(params["N2"])
    ) + h_epsilon * h_gamma * (t[:, None, None] / alpha[None, :, :] ** 2)

    g = torch.exp(
        -0.5 * ((torch.tensor(f, device=device) - params["f0"]) / params["BW"]) ** 2
    ) * torch.cos(2 * torch.pi * params["BSM"] * (torch.tensor(f, device=device) - params["f0"]))

    return {
        "G": g.detach().cpu().numpy(),
        "H": (h * gain[None, :, :]).detach().cpu().numpy(),
        "H_alpha_deriv": (h_alpha_deriv * gain[None, :, :]).detach().cpu().numpy(),
        "gain": gain.detach().cpu().numpy(),
    }


def compute_convolution(args, strf, target_fft):
    rate_params = args["strf"].get("rate_params", {})
    legacy_params = args["strf"].get("legacy_match", {})
    stim_gain = rate_params.get("stimGain", 0.5)
    mean_rate = rate_params.get("mean_rate", 0.1)

    drive = stim_gain * np.einsum("ft,f->t", target_fft["spec"], strf["G"])
    frate = fftconvolve(drive[:, None, None], strf["H"], mode="full", axes=0)
    frate = mean_rate * frate[: len(target_fft["t"])]

    frate_alpha_deriv = fftconvolve(drive[:, None, None], strf["H_alpha_deriv"], mode="full", axes=0)
    frate_alpha_deriv = mean_rate * frate_alpha_deriv[: len(target_fft["t"])]
    if legacy_params.get("alpha_derivative_units", "milliseconds") == "seconds":
        frate_alpha_deriv = frate_alpha_deriv * 1000.0

    crop_start = int(args["strf"]["strf_params"].get("maxdelay", strf["H"].shape[0]))
    return {
        "frate": frate[crop_start:],
        "frate_alpha_deriv": frate_alpha_deriv[crop_start:],
        "gain": strf["gain"],
    }


def compute_rates(args, states, rates, device):
    frate = rates["frate"].copy()
    frate_alpha_deriv = rates["frate_alpha_deriv"].copy()
    gain = rates["gain"][None, :, :]
    safe_gain = np.maximum(gain, 1e-12)

    onset_rate = frate.copy()
    onset_rate[onset_rate < 0] = 0
    onset_rate_deriv = frate_alpha_deriv.copy()
    onset_rate_deriv[onset_rate_deriv < 0] = 0

    offset_rate = -frate + np.max(frate, axis=0, keepdims=True) * 0.75
    offset_rate_deriv = -frate_alpha_deriv + np.max(frate_alpha_deriv, axis=0, keepdims=True) * 0.3
    for batch_index in range(offset_rate.shape[1]):
        for cell_index in range(offset_rate.shape[2]):
            nonpositive = np.where(offset_rate[:, batch_index, cell_index] <= 0)[0]
            if nonpositive.size:
                first_nonpositive = nonpositive[0]
                offset_rate[:first_nonpositive, batch_index, cell_index] = 0
                offset_rate_deriv[:first_nonpositive, batch_index, cell_index] = 0

    offset_rate[offset_rate < 0] = 0
    offset_rate_deriv[offset_rate_deriv < 0] = 0

    onset_rate_gain_deriv = onset_rate / safe_gain
    offset_rate_gain_deriv = offset_rate / safe_gain

    return {
        "onset_rate": torch.tensor(onset_rate, dtype=torch.float32, device=device),
        "offset_rate": torch.tensor(offset_rate, dtype=torch.float32, device=device),
        "onset_rate_deriv": torch.tensor(onset_rate_deriv, dtype=torch.float32, device=device),
        "offset_rate_deriv": torch.tensor(offset_rate_deriv, dtype=torch.float32, device=device),
        "onset_rate_gain_deriv": torch.tensor(onset_rate_gain_deriv, dtype=torch.float32, device=device),
        "offset_rate_gain_deriv": torch.tensor(offset_rate_gain_deriv, dtype=torch.float32, device=device),
    }
