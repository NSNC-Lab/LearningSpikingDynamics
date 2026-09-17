import torch


def relative_terms(delta, a, b, c):
    """Bounded relative-spike probability and its parameter derivatives."""
    tanh_u = torch.tanh(a * delta - b)
    shape = 0.5 * (1 + tanh_u)
    raw = c * shape
    probability = raw.clamp(0, 1)
    active = ((raw > 0) & (raw < 1)).to(raw.dtype)
    slope = 0.5 * (1 - tanh_u.square()) * active
    return probability, c * delta * slope, -c * slope, shape * active


def absolute_release(delta, abs_ref, dt):
    """Smooth absolute-refractory release in timestep-index units."""
    tanh_u = torch.tanh(delta - abs_ref / dt)
    release = 0.5 * (1 + tanh_u)
    derivative = -0.5 * (1 - tanh_u.square()) / dt
    return release, derivative


def rate_terms(sim_count, target_count, exposure, weight):
    sim_rate, target_rate = sim_count / exposure, target_count / exposure
    error = sim_rate - target_rate
    return weight * error.square(), 2 * weight * error / exposure


def cv_from_stats(n, total, total_sq, dtotal, dtotal_sq, eps=1e-8):
    """Sample CV and dCV/dtheta from pooled within-trial ISI statistics."""
    safe_n = n.clamp_min(2)
    mean = total / safe_n
    variance = (total_sq - total.square() / safe_n) / (safe_n - 1)
    valid = (n > 1) & (mean > eps) & (variance > eps)
    std = variance.clamp_min(eps).sqrt()
    dmean = dtotal / safe_n[..., None]
    dvariance = (dtotal_sq - 2 * total[..., None] * dtotal / safe_n[..., None]) / (safe_n - 1)[..., None]
    cv = torch.where(valid, std / mean.clamp_min(eps), torch.zeros_like(mean))
    dcv = dvariance / (2 * std[..., None] * mean.clamp_min(eps)[..., None]) - std[..., None] * dmean / mean.clamp_min(eps).square()[..., None]
    return cv, torch.where(valid[..., None], dcv, torch.zeros_like(dcv)), valid
