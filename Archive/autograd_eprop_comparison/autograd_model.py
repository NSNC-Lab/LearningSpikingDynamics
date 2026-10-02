"""Functional hard-spiking forward model with explicit surrogate backward rules."""
from collections import deque

import torch

PARAMETERS = (
    "Strf_gain", "Strf_alpha", "output_ad", "on_ron_gSYN", "off_ron_gSYN",
    "on_sonoff_gSYN", "off_sonoff_gSYN", "sonoff_ron_gSYN", "abs_ref",
    "rel_ref_a", "rel_ref_b", "rel_ref_c",
)


class SurrogateValue(torch.autograd.Function):
    """Use an exact hard value in forward and a supplied smooth graph in backward."""
    @staticmethod
    def forward(ctx, hard, smooth):
        return hard.clone()

    @staticmethod
    def backward(ctx, grad):
        return None, grad


def surrogate(hard, smooth):
    return SurrogateValue.apply(hard, smooth)


class Model:
    def __init__(self, template, parameters, dt, reset_gradient="detach"):
        self.p = parameters
        self.dt = dt
        self.reset_gradient = reset_gradient
        self.ns = template["neurons"]["Static"]
        self.ss = template["synapses"]["Static"]
        self.n = {
            k: {name: value[..., -1].detach().clone()
                for name, value in state.items()
                if name in ("V", "g_ad", "noise_sn", "noise_xn")}
            for k, state in template["neurons"]["Dynamic"].items()
        }
        self.s = {k: {name: value[..., -1].detach().clone()
                      for name, value in state.items() if name.startswith("PSC_")}
                  for k, state in template["synapses"]["Dynamic"].items()}
        self.last = {k: torch.full_like(v["V"], -30) for k, v in self.n.items()}
        self.history = {k: deque(maxlen=max(int(s["PSC_delay"] / dt)
                                           for s in self.ss.values()) + 1)
                        for k in self.n}
        self.spike_times = {k: template["neurons"]["Dynamic"][k]["tspike"].clone()
                            for k in self.n}
        self.buffer = {k: torch.zeros_like(self.last[k], dtype=torch.long) for k in self.n}

    def param(self, name):
        return self.p[name][:, None, :]

    def step(self, t, inputs, rates, noise, uniform):
        dt = self.dt
        for k, d in self.n.items():
            c, v = self.ns[k], d["V"]
            dv = ((c["E_L"] - v) - c["R"] * d["g_ad"] * (v - c["E_k"])) / c["tau"]
            if c["input"]:
                # Linearized rate graph at frozen parameters; no resampling in backward.
                dr = ((self.param("Strf_gain") - self.param("Strf_gain").detach())
                      * rates[f"{k}set_rate_gain_deriv"][t, :, None, :]
                      + (self.param("Strf_alpha") - self.param("Strf_alpha").detach())
                      * rates[f"{k}set_rate_deriv"][t, :, None, :])
                spike = surrogate(inputs[f"{k}set_spks"][..., t], dr * dt / 1000)
                dv = dv - c["R"] * c["g_postIC"] * spike * (v - c["E_exc"]) / c["tau"]
            for name, sc in self.ss.items():
                if name.rsplit("_", 1)[1] == k:
                    dv = dv - c["R"] * self.s[name]["PSC_s"] * self.param(name + "_gSYN") * (v - sc["ESYN"]) / c["tau"]
            if c["noise"]:
                dv = dv - c["R"] * c["nSYN"] * d["noise_sn"] * (v - c["noise_E_exc"]) / c["tau"]
                sn_update = (c["noise_scale"] * d["noise_xn"] - d["noise_sn"]) / c["tauR_N"]
                xn_update = -d["noise_xn"] / c["tauD_N"] + noise[..., t] / dt
                d["noise_sn"] = d["noise_sn"] + sn_update * dt
                d["noise_xn"] = d["noise_xn"] + xn_update * dt
            d["V"] = v + dv * dt
            d["g_ad"] = d["g_ad"] + (-d["g_ad"] / c["tau_ad"]) * dt

        for k, d in self.s.items():
            c = self.ss[k]
            d["PSC_s"] = d["PSC_s"] + dt * (c["scale"] * d["PSC_x"] - d["PSC_s"]) / c["tauR"]
            d["PSC_x"] = d["PSC_x"] + dt * -d["PSC_x"] / c["tauD"]
            d["PSC_F"] = d["PSC_F"] + dt * (1 - d["PSC_F"]) / c["tauF"]
            d["PSC_P"] = d["PSC_P"] + dt * (1 - d["PSC_P"]) / c["tauP"]

        output = None
        for k, d in self.n.items():
            c = self.ns[k]
            ref = self.param("abs_ref") if c["output"] else c["t_ref"]
            active = (t > self.last[k] + ref.detach() / dt) if c["output"] else (t > self.last[k] + ref / dt)
            hard_v = torch.where(active, d["V"].detach(), c["V_reset"])
            v = torch.where(active, d["V"], c["V_reset"])
            if c["output"]:
                # Match the tanh absolute-refractory derivative, in timestep units.
                gate = (1 + torch.tanh((t - self.last[k]) - ref / dt)) / 2
                v = v + (gate - gate.detach()) * (d["V"].detach() - c["V_reset"])
            v = surrogate(hard_v, v)
            q = (1 + torch.tanh((v - c["V_thresh"]) / 5)) / 2
            if c["output"]:
                p = self.param("rel_ref_c") * torch.tanh(self.param("rel_ref_a") * (t - self.last[k]) - self.param("rel_ref_b")) + self.param("rel_ref_c")
                # Root code clips the forward probability but uses unclipped partials.
                probability = surrogate(p.detach().clamp(0, 1), p)
                hard = ((hard_v >= c["V_thresh"]) & (uniform < probability.detach())).to(v.dtype)
                smooth = q * probability * active
            else:
                hard = (hard_v >= c["V_thresh"]).to(v.dtype)
                smooth = q * active
            z = surrogate(hard, smooth)
            self.history[k].append(z)
            # Event time is discrete, as in the original sensitivity implementation.
            with torch.no_grad():
                old = self.spike_times[k].gather(-1, self.buffer[k].unsqueeze(-1))
                self.spike_times[k].scatter_(-1, self.buffer[k].unsqueeze(-1),
                                            torch.where(hard.bool().unsqueeze(-1), t, old))
                self.buffer[k] = (self.buffer[k] + hard.long()) % 5
                self.last[k] = torch.where(hard.bool(), t, self.last[k])
            event = z.detach() if self.reset_gradient == "detach" else z
            reset_v = (1 - event) * v + event * c["V_reset"]
            d["V"] = surrogate(torch.where(hard.bool(), c["V_reset"], hard_v), reset_v)
            increment = self.param("output_ad") if c["output"] else c["g_inc"]
            d["g_ad"] = d["g_ad"] + event * increment
            if c["output"]:
                output = z

        for k, d in self.s.items():
            c = self.ss[k]
            pre = k.split("_")[0]
            delay = int(c["PSC_delay"] / dt)
            hard = (t == (self.spike_times[pre] + delay).long()).any(-1).to(output.dtype)
            smooth = self.history[pre][-delay - 1] if len(self.history[pre]) > delay else torch.zeros_like(hard)
            z = surrogate(hard, smooth)
            q, f, p = d["PSC_q"], d["PSC_F"], d["PSC_P"]
            # x receives the OLD q, exactly as in both current forward implementations.
            d["PSC_x"] = d["PSC_x"] + z * q
            d["PSC_q"] = surrogate(torch.where(hard.bool(), (f * p).detach(), q.detach()), q + z * (f * p - q))
            d["PSC_F"] = f + z * c["PSC_fF"] * (c["PSC_maxF"] - f)
            d["PSC_P"] = surrogate(torch.where(hard.bool(), (p * (1 - c["PSC_fP"])).detach(), p.detach()), p * (1 - z * c["PSC_fP"]))
        return output


def cosine(a, b):
    import numpy as np
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    result = np.full(denom.shape, np.nan)
    np.divide((a * b).sum(-1), denom, out=result, where=denom > 1e-30)
    return np.clip(result, -1, 1)
