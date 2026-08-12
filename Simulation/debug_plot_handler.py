from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def enabled(args):
    return bool(args.get("debug_plots", {}).get("enabled", False))


def _cfg(args):
    return args.get("debug_plots", {})


def _output_dir(args):
    output_dir = Path(_cfg(args).get("output_dir", "Archive/Debugging/live_loss_debug"))
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[1] / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _selected_indices(args, states=None):
    cfg = _cfg(args)
    batch_index = int(cfg.get("batch_index", 0))
    cell_index = int(cfg.get("cell_index", 0))
    batch_count = int(args["simulation"]["batch_size"])
    cell_count = len(args["simulation"]["cell_targets"])
    if states is not None:
        spikes = states["neurons"]["Dynamic"]["ron"]["spikes_holder"]
        batch_count = spikes.shape[0]
        cell_count = spikes.shape[2]
    return min(batch_index, batch_count - 1), min(cell_index, cell_count - 1)


def _time_axis(args, samples):
    return np.arange(samples) * float(args["simulation"]["dt"])


def _binned_sim_psth(spikes, bins, granularity):
    if bins <= 0:
        return np.zeros(0, dtype=np.float32)
    trimmed = spikes[:, : bins * granularity]
    return trimmed.reshape(trimmed.shape[0], bins, granularity).sum(axis=(0, 2))


def _plot_raster(ax, raster, timestep, dt_ms, title, color):
    visible = raster[:, :timestep]
    trial_idx, time_idx = np.where(visible > 0)
    if time_idx.size:
        ax.scatter(time_idx * dt_ms, trial_idx + 1, s=4, color=color, linewidths=0)
    ax.set_ylim(0.5, raster.shape[0] + 0.5)
    ax.invert_yaxis()
    ax.set_ylabel("trial")
    ax.set_title(title)


def plot_input_rates(args, rate_object, epoch):
    if not enabled(args) or not _cfg(args).get("plot_input_rates", True):
        return

    batch_index, cell_index = _selected_indices(args)
    onset_rate = _to_numpy(rate_object["onset_rate"])[:, batch_index, cell_index]
    offset_rate = _to_numpy(rate_object["offset_rate"])[:, batch_index, cell_index]
    time_ms = _time_axis(args, onset_rate.shape[0])

    fig_num = "debug_input_rates"
    fig = plt.figure(fig_num, figsize=(11, 4))
    fig.clf()
    ax = fig.subplots(1, 1)
    ax.plot(time_ms, onset_rate, label="onset", linewidth=1.0)
    ax.plot(time_ms, offset_rate, label="offset", linewidth=1.0, alpha=0.85)
    ax.set_xlim(0, args["simulation"]["sim_len"] * args["simulation"]["dt"])
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("rate (Hz)")
    ax.set_title(f"Epoch {epoch} input rates: batch {batch_index}, cell index {cell_index}")
    ax.legend(frameon=False)
    fig.tight_layout()
    _emit_plot(args, fig, f"epoch_{epoch:03d}_input_rates.png")


def plot_loss_update(args, states, gt_data, timestep, epoch):
    if not enabled(args):
        return

    granularity = int(args["simulation"]["PSTH_granularity"])
    update_index = timestep // granularity
    every = int(_cfg(args).get("plot_every_n_loss_updates", 1))
    if every > 1 and update_index % every != 0:
        return
    _plot_progress(args, states, gt_data, timestep, epoch, f"loss update {update_index}", f"epoch_{epoch:03d}_step_{timestep:05d}.png")


def plot_epoch_summary(args, states, gt_data, epoch):
    if not enabled(args) or not _cfg(args).get("plot_epoch_summary", True):
        return
    _plot_progress(args, states, gt_data, int(args["simulation"]["sim_len"]), epoch, "epoch summary", f"epoch_{epoch:03d}_summary.png")


def _plot_progress(args, states, gt_data, timestep, epoch, title_suffix, filename):
    batch_index, cell_index = _selected_indices(args, states)
    dt_ms = float(args["simulation"]["dt"])
    sim_len = int(args["simulation"]["sim_len"])
    granularity = int(args["simulation"]["PSTH_granularity"])
    safe_timestep = min(int(timestep), sim_len)
    bins = min(safe_timestep // granularity, _to_numpy(gt_data["psth_holder"]).shape[1])

    sim_raster = _to_numpy(states["neurons"]["Dynamic"]["ron"]["spikes_holder"][batch_index, :, cell_index, :])
    data_raster = _to_numpy(gt_data["raster_holder"][cell_index, :, :])
    sim_psth = _binned_sim_psth(sim_raster, bins, granularity)
    data_psth = _to_numpy(gt_data["psth_holder"][cell_index, :bins])
    bin_time = np.arange(bins) * granularity * dt_ms

    fig_num = "debug_loss_progress"
    fig = plt.figure(fig_num, figsize=(11, 8))
    fig.clf()
    axes = fig.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]})

    if bins:
        axes[0].step(bin_time, data_psth, where="post", label="data PSTH", linewidth=1.2)
        axes[0].step(bin_time, sim_psth, where="post", label="sim PSTH", linewidth=1.1)
    axes[0].axvline(safe_timestep * dt_ms, color="0.65", linewidth=0.8)
    axes[0].set_ylabel("spikes/bin")
    axes[0].set_title(f"Epoch {epoch} {title_suffix}: batch {batch_index}, cell index {cell_index}")
    axes[0].legend(frameon=False, loc="upper right")

    _plot_raster(axes[1], sim_raster, safe_timestep, dt_ms, "sim raster", "#1f77b4")
    _plot_raster(axes[2], data_raster, safe_timestep, dt_ms, "data raster", "#111111")
    axes[2].set_xlabel("time (ms)")

    for ax in axes:
        ax.set_xlim(0, sim_len * dt_ms)

    fig.tight_layout()
    _emit_plot(args, fig, filename)


def _emit_plot(args, fig, filename):
    cfg = _cfg(args)
    if cfg.get("save", True):
        fig.savefig(_output_dir(args) / filename, dpi=int(cfg.get("dpi", 120)))
    if cfg.get("live", True):
        plt.ion()
        fig.canvas.draw_idle()
        plt.pause(float(cfg.get("pause_seconds", 0.001)))
    else:
        plt.close(fig)
