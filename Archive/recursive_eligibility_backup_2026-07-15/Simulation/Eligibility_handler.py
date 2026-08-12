import torch

from Simulation import conditional_handler, ode_handler


def _advance_input_strf_sensitivity(
    args,
    neuron_static,
    neuron_dynamic,
    rate_derivative,
    probability_active,
    param_name,
):
    """Advance one input neuron's persistent dV/dSTRF trace."""
    trace = neuron_dynamic[f"dV_STRF_{param_name}"]
    old_sensitivity = trace[:,:,:,-1]
    active_rate_derivative = (
        rate_derivative[:, None, :]
        * probability_active[:, None, :]
    )
    source = ode_handler.input_rate_voltage_source(
        neuron_static,
        neuron_dynamic['V'][:,:,:,-2],
        active_rate_derivative,
        args['simulation']['dt'],
    )
    trace[:,:,:,-2] = old_sensitivity
    trace[:,:,:,-1] = ode_handler.advance_voltage_sensitivity(
        old_sensitivity,
        neuron_dynamic['input_V_jacobian'],
        source,
        args['simulation']['dt'],
    )

def update_eligibility(args, states, rate_object, timestep):

    # Normalized derivative of the smooth threshold gate
    #   h(V) = 0.5 * (1 + tanh((V - V_thresh) / width)).
    # The recurrent hard events remain frozen in this estimator; psi supplies
    # only the local conditional-mean spike derivative.
    surrogate_width = float(args.get("eligibility", {}).get("surrogate_width_mv", 1.0))
    for k in list(states['neurons']['Static']):
        voltage_offset = (states['neurons']['Dynamic'][k]['V'][:,:,:,-1] - states['neurons']['Static'][k]['V_thresh']) / surrogate_width
        states['neurons']['Static'][k]['psi'] = 0.5 * (1 - torch.tanh(voltage_offset)**2) / surrogate_width

    ron_static = states['neurons']['Static']['ron']
    ron_dynamic = states['neurons']['Dynamic']['ron']
    relative_ref = conditional_handler.refractory_state(states, timestep)
    ron_active = (~conditional_handler.absolute_refractory_mask(
        args, states, "ron", timestep
    )).to(ron_dynamic['V'].dtype)
    # Marginalize the output Bernoulli refractory draw: the local smooth
    # conditional mean is active * relative_probability * h(V).  Input
    # neurons have no corresponding relative-refractory probability factor.
    ron_output_psi = (
        ron_static['psi']
        * ron_active
        * relative_ref["probability"].to(ron_dynamic['V'].dtype)
    )

    # Conductance eligibilities are the output neuron's recursive voltage
    # sensitivities, accumulated over exactly the current PSTH bin.
    for syn_name in list(states['synapses']['Static']):
        if syn_name in ("on_ron", "off_ron", "sonoff_ron"):
            output_dv = states['synapses']['Dynamic'][syn_name]['dV_gSYN'][:,:,:,-1]
        else:
            output_dv = states['synapses']['Dynamic']['sonoff_ron'][f"dV_{syn_name}_gSYN"][:,:,:,-1]
        states['synapses']['Learnable'][syn_name]['gSYN_accum'] = (
            states['synapses']['Learnable'][syn_name]['gSYN_accum']
            + torch.sum(ron_output_psi * output_dv, dim=1)
        )

    #Calculate strf eligibilities
    onset_static = states['neurons']['Static']['on']
    onset_dynamic = states['neurons']['Dynamic']['on']
    offset_static = states['neurons']['Static']['off']
    offset_dynamic = states['neurons']['Dynamic']['off']
    onset_probability = rate_object['onset_rate'][timestep] * args['simulation']['dt']/1000
    offset_probability = rate_object['offset_rate'][timestep] * args['simulation']['dt']/1000
    onset_probability_active = (onset_probability > 0.0) & (onset_probability < 1.0)
    offset_probability_active = (offset_probability > 0.0) & (offset_probability < 1.0)

    states['neurons']['Learnable']['STRF_gain_accum'] = states['neurons']['Learnable']['STRF_gain_accum'] + torch.sum(ron_output_psi*(states['synapses']['Dynamic']['on_ron']['dV_gain'][:,:,:,-1] + states['synapses']['Dynamic']['off_ron']['dV_gain'][:,:,:,-1] + states['synapses']['Dynamic']['sonoff_ron']['dV_gain'][:,:,:,-1]),axis=1)
    states['neurons']['Learnable']['STRF_alpha_accum'] = states['neurons']['Learnable']['STRF_alpha_accum'] + torch.sum(ron_output_psi*(states['synapses']['Dynamic']['on_ron']['dV_alpha'][:,:,:,-1] + states['synapses']['Dynamic']['off_ron']['dV_alpha'][:,:,:,-1] + states['synapses']['Dynamic']['sonoff_ron']['dV_alpha'][:,:,:,-1]),axis=1)

    _advance_input_strf_sensitivity(
        args,
        onset_static,
        onset_dynamic,
        rate_object['onset_rate_gain_deriv'][timestep],
        onset_probability_active,
        "gain",
    )
    _advance_input_strf_sensitivity(
        args,
        onset_static,
        onset_dynamic,
        rate_object['onset_rate_deriv'][timestep],
        onset_probability_active,
        "alpha",
    )
    _advance_input_strf_sensitivity(
        args,
        offset_static,
        offset_dynamic,
        rate_object['offset_rate_gain_deriv'][timestep],
        offset_probability_active,
        "gain",
    )
    _advance_input_strf_sensitivity(
        args,
        offset_static,
        offset_dynamic,
        rate_object['offset_rate_deriv'][timestep],
        offset_probability_active,
        "alpha",
    )

    onset_active = (~conditional_handler.absolute_refractory_mask(
        args, states, "on", timestep
    )).to(onset_dynamic['V'].dtype)
    offset_active = (~conditional_handler.absolute_refractory_mask(
        args, states, "off", timestep
    )).to(offset_dynamic['V'].dtype)
    onset_pre_derivs = {
        param_name: onset_static['psi'] * onset_active * onset_dynamic[f"dV_STRF_{param_name}"][:,:,:,-1]
        for param_name in ("gain", "alpha")
    }
    offset_pre_derivs = {
        param_name: offset_static['psi'] * offset_active * offset_dynamic[f"dV_STRF_{param_name}"][:,:,:,-1]
        for param_name in ("gain", "alpha")
    }

    local_pre_derivs = {
        "on_ron": {
            "gain": onset_pre_derivs["gain"],
            "alpha": onset_pre_derivs["alpha"],
        },
        "off_ron": {
            "gain": offset_pre_derivs["gain"],
            "alpha": offset_pre_derivs["alpha"],
        },
        "on_sonoff": {
            "gain": onset_pre_derivs["gain"],
            "alpha": onset_pre_derivs["alpha"],
        },
        "off_sonoff": {
            "gain": offset_pre_derivs["gain"],
            "alpha": offset_pre_derivs["alpha"],
        }
    }
    # Preserve the first-layer estimator explicitly as a fixed-grid soft event
    # strength.  Delivery and the complete STP tangent event map occur beside
    # the matching hard forward event in conditional_handler.condtion3.
    for syn_name in ("on_ron", "off_ron", "on_sonoff", "off_sonoff"):
        delay_steps = int(states['synapses']['Static'][syn_name]['PSC_delay']/args['simulation']['dt'])
        delay_idx = timestep % (delay_steps + 1)
        for param_name in ("gain", "alpha"):
            syn_dynamic = states['synapses']['Dynamic'][syn_name]
            syn_dynamic[f"dsoft_{param_name}_delay"][:,:,:,delay_idx] = (
                local_pre_derivs[syn_name][param_name]
            )
    
    #Calculate adaptation eligibilities
    states['neurons']['Learnable']['ron']["output_ad_accum"] = (
        states['neurons']['Learnable']['ron']["output_ad_accum"]
        + torch.sum(
            ron_output_psi * ron_dynamic['dV_output_ad'][:,:,:,-1],
            dim=1,
        )
    )
    
    #Calculate refractoriness eligibilities
    elapsed_steps = relative_ref["elapsed_steps"]
    has_previous_spike = relative_ref["has_previous_spike"].to(ron_dynamic['V'].dtype)
    abs_argument = elapsed_steps - states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:]/args['simulation']['dt']
    recovered_gate = torch.where(
        relative_ref["has_previous_spike"],
        0.5*(1+torch.tanh(abs_argument)),
        torch.ones_like(abs_argument),
    )
    states["neurons"]["Learnable"]["ron"]["abs_ref_accum"] = states["neurons"]["Learnable"]["ron"]["abs_ref_accum"] + torch.sum(ron_static['psi']*(ron_dynamic['V'][:,:,:,-1] - ron_static['V_reset'])*(-0.5/args['simulation']['dt']*(1-torch.tanh(abs_argument)**2))*has_previous_spike,axis=1)

    voltage_gate = 0.5*(1+torch.tanh((ron_dynamic['V'][:,:,:,-1]-ron_static['V_thresh'])/surrogate_width))
    hazard_argument = states["neurons"]["Learnable"]["ron"]["rel_ref_a"][:,None,:]*elapsed_steps-states["neurons"]["Learnable"]["ron"]["rel_ref_b"][:,None,:]
    hazard_sech2 = 1-torch.tanh(hazard_argument)**2
    clamp_active = relative_ref["clamp_active"].to(ron_dynamic['V'].dtype)
    common_gate = voltage_gate*recovered_gate*clamp_active
    states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum"] = states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum"] + torch.sum(common_gate*states["neurons"]["Learnable"]["ron"]["rel_ref_c"][:,None,:]*elapsed_steps*hazard_sech2,axis=1)
    states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum"] = states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum"] + torch.sum(common_gate*-states["neurons"]["Learnable"]["ron"]["rel_ref_c"][:,None,:]*hazard_sech2,axis=1)
    states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum"] = states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum"] + torch.sum(common_gate*(torch.tanh(hazard_argument)+1),axis=1)

    return states
