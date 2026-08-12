import torch


def _valid_spike_times(dynamic):
    """Return the spike-time ring buffer and its valid-entry mask.

    Negative values are initialization sentinels, not biological spikes.  Theyrefractory_state
    must never trigger delayed synapses or refractory clamps.
    """
    spike_times = dynamic["tspike"]
    return spike_times, spike_times >= 0


def refractory_state(states, timestep):
    """Return elapsed time and the output neuron's relative-ref hazard.

    A neuron with no previous spike starts fully recovered.  ``raw_probability``
    and ``clamp_active`` are exposed so eligibility code can differentiate the
    exact same clamped forward hazard.
    """
    dynamic = states["neurons"]["Dynamic"]["ron"]
    learnable = states["neurons"]["Learnable"]["ron"]
    spike_times, valid = _valid_spike_times(dynamic)
    has_previous_spike = torch.any(valid, dim=-1)
    last_spike = spike_times.max(dim=-1).values
    elapsed_steps = timestep - last_spike

    a = learnable["rel_ref_a"][:, None, :]
    b = learnable["rel_ref_b"][:, None, :]
    c = learnable["rel_ref_c"][:, None, :]
    raw_probability = c * (torch.tanh(a * elapsed_steps - b) + 1.0)
    clamp_active = has_previous_spike & (raw_probability > 0.0) & (raw_probability < 1.0)
    probability = torch.where(
        has_previous_spike,
        torch.clamp(raw_probability, min=0.0, max=1.0),
        torch.ones_like(raw_probability),
    )
    return {
        "last_spike": last_spike,
        "elapsed_steps": elapsed_steps,
        "has_previous_spike": has_previous_spike,
        "raw_probability": raw_probability,
        "clamp_active": clamp_active,
        "probability": probability,
    }


def absolute_refractory_mask(args, states, neuron_name, timestep):
    """Return the exact mask that condition1 will clamp this timestep."""
    spike_times, valid = _valid_spike_times(states['neurons']['Dynamic'][neuron_name])
    if states['neurons']['Static'][neuron_name]['output'] == 1:
        refractory_duration = (
            states['neurons']['Learnable'][neuron_name]['abs_ref'][:, None, :, None]
            / args['simulation']['dt']
        )
    else:
        refractory_duration = (
            states['neurons']['Static'][neuron_name]['t_ref']
            / args['simulation']['dt']
        )
    return torch.any(valid & (timestep <= spike_times + refractory_duration), dim=-1)


def _zero_trace_at_indices(trace, previous_indices, current_indices):
    """Reset one trace using indices computed once by the caller."""
    trace[previous_indices] = trace[current_indices]
    trace[current_indices] = 0


def reset_voltage_sensitivities(states, neuron_name, indices):
    """Apply one reset index tuple to every voltage-sensitivity trace.

    The enclosing conditional has already established that the mask is
    non-empty and called ``torch.where``.  Reusing those indices avoids a
    blocking GPU ``.item()`` and another ``torch.where`` for every trace.
    """
    previous_indices = indices + (-2,)
    current_indices = indices + (-1,)

    if neuron_name == "ron":
        for syn_name in ("on_ron", "off_ron", "sonoff_ron"):
            dynamic = states['synapses']['Dynamic'][syn_name]
            for trace_name in ("dV_gain", "dV_alpha", "dV_gSYN"):
                if trace_name in dynamic:
                    _zero_trace_at_indices(
                        dynamic[trace_name], previous_indices, current_indices
                    )

        indirect_dynamic = states['synapses']['Dynamic']['sonoff_ron']
        for source_name in ("on_sonoff", "off_sonoff"):
            _zero_trace_at_indices(
                indirect_dynamic[f"dV_{source_name}_gSYN"],
                previous_indices,
                current_indices,
            )

        _zero_trace_at_indices(
            states['neurons']['Dynamic']['ron']['dV_output_ad'],
            previous_indices,
            current_indices,
        )

    elif neuron_name == "sonoff":
        for syn_name in ("on_sonoff", "off_sonoff"):
            dynamic = states['synapses']['Dynamic'][syn_name]
            for trace_name in ("dV_gain", "dV_alpha", "dV_gSYN"):
                if trace_name in dynamic:
                    _zero_trace_at_indices(
                        dynamic[trace_name], previous_indices, current_indices
                    )

    elif neuron_name in ("on", "off"):
        dynamic = states['neurons']['Dynamic'][neuron_name]
        for param_name in ("gain", "alpha"):
            _zero_trace_at_indices(
                dynamic[f"dV_STRF_{param_name}"],
                previous_indices,
                current_indices,
            )


def soft_event_stp_tangent(
    forward_F,
    forward_P,
    old_dF,
    old_dP,
    old_dx,
    old_ds,
    event_mask,
    event_strength_derivative,
    synapse_static,
):
    """Differentiate a fixed-grid soft event-strength relaxation.

    ``event_mask`` is the hard forward event u in {0, 1}, while
    ``event_strength_derivative`` is the straight-through du/dtheta.  There is
    no event-time saltation in this estimator, so s remains continuous.
    """
    u = event_mask.to(forward_F.dtype)
    w = event_strength_derivative.to(forward_F.dtype)
    event_q = forward_F * forward_P
    d_event_q = forward_P * old_dF + forward_F * old_dP
    facilitation = synapse_static['PSC_fF']
    depression = synapse_static['PSC_fP']

    new_dF = (
        (1.0 - u * facilitation) * old_dF
        + facilitation * (synapse_static['PSC_maxF'] - forward_F) * w
    )
    new_dP = (
        (1.0 - u * depression) * old_dP
        - depression * forward_P * w
    )
    new_dx = old_dx + u * d_event_q + event_q * w
    return new_dF, new_dP, new_dx, old_ds, d_event_q


def time_shift_stp_tangent(
    forward_F,
    forward_P,
    old_dF,
    old_dP,
    old_dx,
    old_ds,
    event_mask,
    event_time_derivative,
    synapse_static,
):
    """Apply the exact saltation for a parameter-dependent event time.

    ``event_time_derivative`` is raw dt_event/dtheta in milliseconds per
    parameter.  Static axonal delay does not alter it.
    """
    u = event_mask.to(forward_F.dtype)
    zeta = event_time_derivative.to(forward_F.dtype) * u
    event_q = forward_F * forward_P
    recovery_F = (1.0 - forward_F) / synapse_static['tauF']
    recovery_P = (1.0 - forward_P) / synapse_static['tauP']
    d_event_q = (
        u * (forward_P * old_dF + forward_F * old_dP)
        + (forward_P * recovery_F + forward_F * recovery_P) * zeta
    )
    facilitation = synapse_static['PSC_fF']
    depression = synapse_static['PSC_fP']

    new_dF = (
        (1.0 - u * facilitation) * old_dF
        + facilitation
        * (synapse_static['PSC_maxF'] - 1.0)
        * zeta
        / synapse_static['tauF']
    )
    new_dP = (
        (1.0 - u * depression) * old_dP
        - depression * zeta / synapse_static['tauP']
    )
    new_dx = old_dx + d_event_q + event_q * zeta / synapse_static['tauD']
    new_ds = (
        old_ds
        - synapse_static['scale'] * event_q * zeta / synapse_static['tauR']
    )
    return new_dF, new_dP, new_dx, new_ds, d_event_q


def _write_stp_tangent(dynamic, trace_name, tangent):
    new_dF, new_dP, new_dx, new_ds, _ = tangent
    dynamic[f"dPSC_F_{trace_name}"][:,:,:,-1] = new_dF
    dynamic[f"dPSC_P_{trace_name}"][:,:,:,-1] = new_dP
    dynamic[f"dPSC_x_{trace_name}"][:,:,:,-1] = new_dx
    dynamic[f"dPSC_s_{trace_name}"][:,:,:,-1] = new_ds


def run_conditionals(args, states,timestep):
    
    states = condtion1(args, states,timestep) #Previously condition 2b   (Absolute refractory period)
    states = condtion2(args, states,timestep) #Previously condtion 1 and 2a     (Register Spike) (Reset Voltage and adaptation)
    states = condtion3(args, states,timestep) #Previously condtion 3     (Update PSCs)
    
    return states


def condtion1(args, states,timestep):
    for k in list(states['neurons']['Static']):
        mask = absolute_refractory_mask(args, states, k, timestep)
        if  torch.any(mask).item():
            spikers = torch.where(mask)
            states['neurons']['Dynamic'][k]['V'][spikers + (-2,)] = states['neurons']['Dynamic'][k]['V'][spikers + (-1,)] 
            states['neurons']['Dynamic'][k]['V'][spikers + (-1,)] = states['neurons']['Static'][k]['V_reset']
            reset_voltage_sensitivities(states, k, spikers)
    return states

def condtion2(args, states,timestep):
    # A time-ring slot is reused after delay_steps + 1 timesteps.  Clear it
    # before recording any sonoff spike-time derivatives for this timestep.
    sonoff_ron_dynamic = states['synapses']['Dynamic']['sonoff_ron']
    sonoff_ron_static = states['synapses']['Static']['sonoff_ron']
    sonoff_delay_steps = int(
        sonoff_ron_static['PSC_delay'] / args['simulation']['dt']
    )
    sonoff_write_idx = timestep % (sonoff_delay_steps + 1)
    for trace_name in (
        "gain",
        "alpha",
        "on_sonoff_gSYN",
        "off_sonoff_gSYN",
    ):
        sonoff_ron_dynamic[f"dtime_{trace_name}_delay"][:,:,:,sonoff_write_idx] = 0

    for k in list(states['neurons']['Static']):
        if states['neurons']['Static'][k]['output'] == 1:
            relative_ref = refractory_state(states, timestep)
            mask = ((states['neurons']['Dynamic'][k]['V'][:,:,:,-1] >= states['neurons']['Static'][k]['V_thresh']) & (torch.rand(relative_ref["probability"].shape,device=torch.device(args['simulation']['device'])) < relative_ref["probability"])).to(torch.int64)
            states['neurons']['Dynamic'][k]["spikes_holder"][:,:,:,timestep] = mask
        else:
            mask = ((states['neurons']['Dynamic'][k]['V'][:,:,:,-1] >= states['neurons']['Static'][k]['V_thresh'])).to(torch.int64)
        states['neurons']['Dynamic'][k]['spike_mask'].copy_(mask.bool())
        if torch.any(mask).item():
            spikers = torch.where(mask)
            states['neurons']['Dynamic'][k]['tspike'][spikers + (states['neurons']['Dynamic'][k]['buffer_index'][spikers].to(torch.int64)-1,)] = timestep
            states['neurons']['Dynamic'][k]['buffer_index'][spikers] = states['neurons']['Dynamic'][k]['buffer_index'][spikers] % 5 + 1
            states['neurons']['Dynamic'][k]['V'][spikers + (-2,)] = states['neurons']['Dynamic'][k]['V'][spikers + (-1,)]
            states['neurons']['Dynamic'][k]['V'][spikers + (-1,)] = states['neurons']['Static'][k]['V_reset']
            states['neurons']['Dynamic'][k]['g_ad'][spikers + (-2,)] = states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)]
            if states['neurons']['Static'][k]['output'] == 1:
                states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)] = states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)] + states['neurons']['Learnable'][k]["output_ad"][spikers[0],None,spikers[2]].T
                output_ad_sensitivity = states['neurons']['Dynamic'][k]['dg_ad_output_ad']
                output_ad_sensitivity[spikers + (-2,)] = output_ad_sensitivity[spikers + (-1,)]
                output_ad_sensitivity[spikers + (-1,)] = output_ad_sensitivity[spikers + (-1,)] + 1
                reset_voltage_sensitivities(states, k, spikers)
            else:
                if k == "sonoff":
                    v_slope = states['neurons']['Dynamic'][k].get('V_slope', (states['neurons']['Dynamic'][k]['V'][:,:,:,-1] - states['neurons']['Dynamic'][k]['V'][:,:,:,-2]) / args['simulation']['dt'])[spikers]
                    v_slope_sign = torch.where(v_slope < 0, -torch.ones_like(v_slope), torch.ones_like(v_slope))
                    v_slope = torch.where(torch.abs(v_slope) < 1e-6, v_slope_sign * 1e-6, v_slope)
                    sonoff_ron_dynamic = states['synapses']['Dynamic']['sonoff_ron']
                    sonoff_ron_static = states['synapses']['Static']['sonoff_ron']
                    delay_steps = int(sonoff_ron_static['PSC_delay']/args['simulation']['dt'])
                    delay_idx = timestep % (delay_steps + 1)
                    for param_name in ("gain", "alpha"):
                        dV_param = states['synapses']['Dynamic']['on_sonoff'][f"dV_{param_name}"][spikers + (-1,)] + states['synapses']['Dynamic']['off_sonoff'][f"dV_{param_name}"][spikers + (-1,)]
                        spike_time_deriv = -dV_param / v_slope
                        sonoff_ron_dynamic[f"dtime_{param_name}_delay"][spikers + (delay_idx,)] += spike_time_deriv

                    for source_name in ("on_sonoff", "off_sonoff"):
                        trace_name = f"{source_name}_gSYN"
                        dV_param = states['synapses']['Dynamic'][source_name]['dV_gSYN'][spikers + (-1,)]
                        spike_time_deriv = -dV_param / v_slope
                        sonoff_ron_dynamic[f"dtime_{trace_name}_delay"][spikers + (delay_idx,)] += spike_time_deriv

                    reset_voltage_sensitivities(states, k, spikers)
                else:
                    reset_voltage_sensitivities(states, k, spikers)
                states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)] = states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)] + states['neurons']['Static'][k]['g_inc']
    
    return states

def condtion3(args, states,timestep):
    for k in list(states['synapses']['Static']):
        pre_neuron_dynamic = states['neurons']['Dynamic'][k.split("_")[0]]
        synapse_static = states['synapses']['Static'][k]
        synapse_dynamic = states['synapses']['Dynamic'][k]
        delay_steps = int(synapse_static['PSC_delay']/args['simulation']['dt'])
        spike_queue = synapse_dynamic['spike_delay_queue']
        queue_size = delay_steps + 1
        write_idx = timestep % queue_size
        spike_queue[:,:,:,write_idx] = pre_neuron_dynamic['spike_mask']

        read_idx = (timestep - delay_steps) % (delay_steps + 1)
        mask = (
            spike_queue[:,:,:,read_idx].clone()
            if timestep >= delay_steps
            else torch.zeros_like(pre_neuron_dynamic['spike_mask'])
        )
        if k == "sonoff_ron":
            for trace_name in (
                "gain",
                "alpha",
                "on_sonoff_gSYN",
                "off_sonoff_gSYN",
            ):
                zeta = (
                    synapse_dynamic[f"dtime_{trace_name}_delay"][:,:,:,read_idx]
                    if timestep >= delay_steps
                    else torch.zeros_like(synapse_dynamic['PSC_F'][:,:,:,-1])
                )
                tangent = time_shift_stp_tangent(
                    synapse_dynamic['PSC_F'][:,:,:,-1],
                    synapse_dynamic['PSC_P'][:,:,:,-1],
                    synapse_dynamic[f"dPSC_F_{trace_name}"][:,:,:,-1],
                    synapse_dynamic[f"dPSC_P_{trace_name}"][:,:,:,-1],
                    synapse_dynamic[f"dPSC_x_{trace_name}"][:,:,:,-1],
                    synapse_dynamic[f"dPSC_s_{trace_name}"][:,:,:,-1],
                    mask,
                    zeta,
                    synapse_static,
                )
                _write_stp_tangent(synapse_dynamic, trace_name, tangent)
        else:
            for trace_name in ("gain", "alpha"):
                event_strength_derivative = (
                    synapse_dynamic[f"dsoft_{trace_name}_delay"][:,:,:,read_idx]
                    if timestep >= delay_steps
                    else torch.zeros_like(synapse_dynamic['PSC_F'][:,:,:,-1])
                )
                tangent = soft_event_stp_tangent(
                    synapse_dynamic['PSC_F'][:,:,:,-1],
                    synapse_dynamic['PSC_P'][:,:,:,-1],
                    synapse_dynamic[f"dPSC_F_{trace_name}"][:,:,:,-1],
                    synapse_dynamic[f"dPSC_P_{trace_name}"][:,:,:,-1],
                    synapse_dynamic[f"dPSC_x_{trace_name}"][:,:,:,-1],
                    synapse_dynamic[f"dPSC_s_{trace_name}"][:,:,:,-1],
                    mask,
                    event_strength_derivative,
                    synapse_static,
                )
                _write_stp_tangent(synapse_dynamic, trace_name, tangent)

        if torch.any(mask).item():
            spikers = torch.where(mask)
            synapse_dynamic['PSC_x'][spikers + (-2,)] = synapse_dynamic['PSC_x'][spikers + (-1,)]
            synapse_dynamic['PSC_q'][spikers + (-2,)] = synapse_dynamic['PSC_q'][spikers + (-1,)]
            synapse_dynamic['PSC_F'][spikers + (-2,)] = synapse_dynamic['PSC_F'][spikers + (-1,)]
            synapse_dynamic['PSC_P'][spikers + (-2,)] = synapse_dynamic['PSC_P'][spikers + (-1,)]
            # Use the recovered F/P state for this event.  PSC_q otherwise
            # retains the previous event's amplitude and is one spike stale.
            event_q = synapse_dynamic['PSC_F'][spikers + (-1,)] * synapse_dynamic['PSC_P'][spikers + (-1,)]
            synapse_dynamic['PSC_x'][spikers + (-1,)] = synapse_dynamic['PSC_x'][spikers + (-1,)] + event_q
            synapse_dynamic['PSC_q'][spikers + (-1,)] = event_q
            synapse_dynamic['PSC_F'][spikers + (-1,)] = synapse_dynamic['PSC_F'][spikers + (-1,)] + synapse_static['PSC_fF']*(synapse_static['PSC_maxF']-synapse_dynamic['PSC_F'][spikers + (-1,)])
            synapse_dynamic['PSC_P'][spikers + (-1,)] = synapse_dynamic['PSC_P'][spikers + (-1,)] * (1-synapse_static['PSC_fP'])
    return states
