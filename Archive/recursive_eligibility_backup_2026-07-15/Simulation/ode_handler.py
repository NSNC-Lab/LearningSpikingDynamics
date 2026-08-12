import torch


def advance_voltage_sensitivity(old_sensitivity, voltage_jacobian, source, dt):
    """Euler step for a voltage sensitivity using the forward step's old state."""
    return old_sensitivity + dt * (voltage_jacobian * old_sensitivity + source)


def conductance_voltage_source(neuron_static, synapse_static, psc_s, voltage):
    """d(dV/dt)/d(gSYN) at fixed voltage and PSC state."""
    return (
        -neuron_static['R']
        * psc_s
        * (voltage - synapse_static['ESYN'])
        / neuron_static['tau']
    )


def adaptation_voltage_source(neuron_static, voltage, adaptation_sensitivity):
    """Voltage-RHS source induced by d(g_ad)/d(output_ad)."""
    return (
        -neuron_static['R']
        * (voltage - neuron_static['E_k'])
        * adaptation_sensitivity
        / neuron_static['tau']
    )

def abs_ref_voltage_source(neuron_static, voltage, abs_ref_sensitivity):


def advance_decay_sensitivity(old_sensitivity, tau, dt):
    """Euler decay shared by g_ad and its parameter sensitivity."""
    return old_sensitivity + dt * (-old_sensitivity / tau)


def advance_stp_sensitivities(
    old_dF,
    old_dP,
    old_dx,
    old_ds,
    synapse_static,
    dt,
):
    """Advance fixed-time STP tangents between presynaptic events.

    The update uses exactly the old tangent state, matching the forward Euler
    ordering for F, P, x, and s in ``run_odes``.
    """
    new_dF = old_dF + dt * (-old_dF / synapse_static['tauF'])
    new_dP = old_dP + dt * (-old_dP / synapse_static['tauP'])
    new_dx = old_dx + dt * (-old_dx / synapse_static['tauD'])
    new_ds = old_ds + dt * (
        synapse_static['scale'] * old_dx - old_ds
    ) / synapse_static['tauR']
    return new_dF, new_dP, new_dx, new_ds


def input_voltage_jacobian(neuron_static, g_ad, input_spikes):
    """d(dV/dt)/dV for an input neuron, evaluated at the old Euler state."""
    return (
        -1.0
        - neuron_static['R'] * g_ad
        - neuron_static['R'] * neuron_static['g_postIC'] * input_spikes
    ) / neuron_static['tau']


def input_rate_voltage_source(neuron_static, voltage, rate_derivative, dt):
    """Voltage-RHS source from d Bernoulli input probability / d parameter.

    ``rate_derivative`` is in Hz per parameter.  The sampled input spike is
    held fixed in the forward pass while its straight-through sensitivity is
    ``dt / 1000 * rate_derivative``.
    """
    return (
        -neuron_static['R']
        * neuron_static['g_postIC']
        * (voltage - neuron_static['E_exc'])
        * (dt / 1000.0)
        * rate_derivative
        / neuron_static['tau']
    )


def run_odes(args, states, pre_processed_spike_object,timestep):
    
    #Run Neuron ODES
    
    for k in list(states['neurons']['Static']):

        #Update Voltage
        if timestep == 0:
            states['neurons']['Static'][k]['projections'] = caluclate_projections(states,k)

        cur_dynamic = states['neurons']['Dynamic'][k]
        cur_static = states['neurons']['Static'][k]

        shared_update_V = ((cur_static['E_L'] - cur_dynamic['V'][:,:,:,-1]) - cur_static['R']*cur_dynamic['g_ad'][:,:,:,-1]*(cur_dynamic['V'][:,:,:,-1]-cur_static['E_k']))/ cur_static['tau']

        if states['neurons']['Static'][k]['input'] == 1:
            spks_string = f"{k}set_spks"    
            input_spikes = pre_processed_spike_object['onset_offset_spks'][spks_string][:,:,:,timestep]
            cur_dynamic['input_V_jacobian'] = input_voltage_jacobian(cur_static,cur_dynamic['g_ad'][:,:,:,-1],input_spikes) #Calculates the d/dV of update holder
            update_holder = shared_update_V - (cur_static['R']*cur_static['g_postIC']*input_spikes*(cur_dynamic['V'][:,:,:,-1]-cur_static['E_exc'])) / cur_static['tau']

        if len(states['neurons']['Static'][k]['projections'])>0:

            update_holder = shared_update_V 

            for m in states['neurons']['Static'][k]['projections']:

                update_holder = update_holder - cur_static['R']*states['synapses']['Dynamic'][m]['PSC_s'][:,:,:,-1]*states['synapses']['Learnable'][m]['gSYN'][:,None,:]*(cur_dynamic['V'][:,:,:,-1]-states['synapses']['Static'][m]['ESYN'])/cur_static['tau']

            if states['neurons']['Static'][k]['noise'] == 1:
            
                update_holder = update_holder - cur_static['R']*cur_static['nSYN']*cur_dynamic['noise_sn'][:,:,:,-1]*(cur_dynamic['V'][:,:,:,-1]-cur_static['noise_E_exc'])/cur_static['tau']
                noise_holder_sn = (cur_static['noise_scale']*cur_dynamic['noise_xn'][:,:,:,-1]-cur_dynamic['noise_sn'][:,:,:,-1])/cur_static['tauR_N']
                noise_holder_xn = -(cur_dynamic['noise_xn'][:,:,:,-1]/cur_static['tauD_N']) + pre_processed_spike_object['noise_spks'][:,:,timestep]/args['simulation']['dt']
                cur_dynamic['noise_sn'][:,:,:,-2] = cur_dynamic['noise_sn'][:,:,:,-1]
                cur_dynamic['noise_xn'][:,:,:,-2] = cur_dynamic['noise_xn'][:,:,:,-1]
                cur_dynamic['noise_sn'][:,:,:,-1] = cur_dynamic['noise_sn'][:,:,:,-1] +noise_holder_sn*args['simulation']['dt']
                cur_dynamic['noise_xn'][:,:,:,-1] = cur_dynamic['noise_xn'][:,:,:,-1] +noise_holder_xn*args['simulation']['dt']

            if k == "ron":
                ron_dV_dV = (-1 - cur_static['R']*cur_dynamic['g_ad'][:,:,:,-1])/cur_static['tau']
                for m in states['neurons']['Static'][k]['projections']:
                    ron_dV_dV = ron_dV_dV - cur_static['R']*states['synapses']['Dynamic'][m]['PSC_s'][:,:,:,-1]*states['synapses']['Learnable'][m]['gSYN'][:,None,:]/cur_static['tau']
                if states['neurons']['Static'][k]['noise'] == 1:
                    # noise_sn[-1] has already been advanced above; the voltage
                    # Euler step used the snapshot retained in [-2].
                    ron_dV_dV = ron_dV_dV - cur_static['R']*cur_static['nSYN']*cur_dynamic['noise_sn'][:,:,:,-2]/cur_static['tau']
                for m in ("on_ron", "off_ron", "sonoff_ron"):
                    for param_name in ("gain", "alpha"):
                        syn_dynamic = states['synapses']['Dynamic'][m]
                        syn_static = states['synapses']['Static'][m]
                        syn_dynamic[f"dV_{param_name}"][:,:,:,-2] = syn_dynamic[f"dV_{param_name}"][:,:,:,-1]
                        syn_dynamic[f"dV_{param_name}"][:,:,:,-1] = syn_dynamic[f"dV_{param_name}"][:,:,:,-1] + args['simulation']['dt']*(ron_dV_dV*syn_dynamic[f"dV_{param_name}"][:,:,:,-1] - cur_static['R']*states['synapses']['Learnable'][m]['gSYN'][:,None,:]*(cur_dynamic['V'][:,:,:,-1]-syn_static['ESYN'])*syn_dynamic[f"dPSC_s_{param_name}"][:,:,:,-1]/cur_static['tau'])

                # Exact old-state recursive sensitivities for conductances that
                # project directly to ron.
                for m in ("on_ron", "off_ron", "sonoff_ron"):
                    syn_dynamic = states['synapses']['Dynamic'][m]
                    old_dv = syn_dynamic['dV_gSYN'][:,:,:,-1]
                    direct_source = conductance_voltage_source(
                        cur_static,
                        states['synapses']['Static'][m],
                        syn_dynamic['PSC_s'][:,:,:,-1],
                        cur_dynamic['V'][:,:,:,-1],
                    )
                    syn_dynamic['dV_gSYN'][:,:,:,-2] = old_dv
                    syn_dynamic['dV_gSYN'][:,:,:,-1] = advance_voltage_sensitivity(
                        old_dv, ron_dV_dV, direct_source, args['simulation']['dt']
                    )

                # Carry on/off -> sonoff conductance sensitivities through the
                # sonoff -> ron inhibitory PSC.
                indirect_dynamic = states['synapses']['Dynamic']['sonoff_ron']
                indirect_static = states['synapses']['Static']['sonoff_ron']
                indirect_gsyn = states['synapses']['Learnable']['sonoff_ron']['gSYN'][:,None,:]
                for source_name in ("on_sonoff", "off_sonoff"):
                    trace_name = f"{source_name}_gSYN"
                    old_dv = indirect_dynamic[f"dV_{trace_name}"][:,:,:,-1]
                    psc_source = (
                        -cur_static['R']
                        * indirect_gsyn
                        * (cur_dynamic['V'][:,:,:,-1] - indirect_static['ESYN'])
                        * indirect_dynamic[f"dPSC_s_{trace_name}"][:,:,:,-1]
                        / cur_static['tau']
                    )
                    indirect_dynamic[f"dV_{trace_name}"][:,:,:,-2] = old_dv
                    indirect_dynamic[f"dV_{trace_name}"][:,:,:,-1] = advance_voltage_sensitivity(
                        old_dv, ron_dV_dV, psc_source, args['simulation']['dt']
                    )

                # Causal output-adaptation sensitivity.  The +1 event term is
                # applied later, only if condition2 accepts an output spike.
                old_dv_output_ad = cur_dynamic['dV_output_ad'][:,:,:,-1]
                old_dgad_output_ad = cur_dynamic['dg_ad_output_ad'][:,:,:,-1]
                cur_dynamic['dV_output_ad'][:,:,:,-2] = old_dv_output_ad
                cur_dynamic['dg_ad_output_ad'][:,:,:,-2] = old_dgad_output_ad
                cur_dynamic['dV_output_ad'][:,:,:,-1] = advance_voltage_sensitivity(
                    old_dv_output_ad,
                    ron_dV_dV,
                    adaptation_voltage_source(
                        cur_static,
                        cur_dynamic['V'][:,:,:,-1],
                        old_dgad_output_ad,
                    ),
                    args['simulation']['dt'],
                )
                cur_dynamic['dg_ad_output_ad'][:,:,:,-1] = advance_decay_sensitivity(
                    old_dgad_output_ad, cur_static['tau_ad'], args['simulation']['dt']
                )


                # Section for the refractoriness sensitivity terms.
                old_dv_abs_ref = cur_dynamic['dV_abs_ref'][:,:,:,-1]
                old_dv_rel_ref_a = cur_dynamic['dV_rel_ref_a'][:,:,:,-1]
                old_dv_rel_ref_b = cur_dynamic['dV_rel_ref_b'][:,:,:,-1]
                old_dv_rel_ref_c = cur_dynamic['dV_rel_ref_c'][:,:,:,-1]

                cur_dynamic['dV_abs_ref'][:,:,:,-2] = old_dv_abs_ref
                cur_dynamic['dV_rel_ref_a'][:,:,:,-2] = old_dv_rel_ref_a
                cur_dynamic['dV_rel_ref_b'][:,:,:,-2] = old_dv_rel_ref_b
                cur_dynamic['dV_rel_ref_c'][:,:,:,-2] = old_dv_rel_ref_c

                cur_dynamic['dV_abs_ref'][:,:,:,-1] = advance_voltage_sensitivity(
                    old_dv_abs_ref,
                    ron_dV_dV,
                    ??,
                    args['simulation']['dt'],
                )


            if k == "sonoff":
                sonoff_dV_dV = (-1 - cur_static['R']*cur_dynamic['g_ad'][:,:,:,-1])/cur_static['tau']
                for m in states['neurons']['Static'][k]['projections']:
                    sonoff_dV_dV = sonoff_dV_dV - cur_static['R']*states['synapses']['Dynamic'][m]['PSC_s'][:,:,:,-1]*states['synapses']['Learnable'][m]['gSYN'][:,None,:]/cur_static['tau']
                for m in ("on_sonoff", "off_sonoff"):
                    for param_name in ("gain", "alpha"):
                        syn_dynamic = states['synapses']['Dynamic'][m]
                        syn_static = states['synapses']['Static'][m]
                        syn_dynamic[f"dV_{param_name}"][:,:,:,-2] = syn_dynamic[f"dV_{param_name}"][:,:,:,-1]
                        syn_dynamic[f"dV_{param_name}"][:,:,:,-1] = syn_dynamic[f"dV_{param_name}"][:,:,:,-1] + args['simulation']['dt']*(sonoff_dV_dV*syn_dynamic[f"dV_{param_name}"][:,:,:,-1] - cur_static['R']*states['synapses']['Learnable'][m]['gSYN'][:,None,:]*(cur_dynamic['V'][:,:,:,-1]-syn_static['ESYN'])*syn_dynamic[f"dPSC_s_{param_name}"][:,:,:,-1]/cur_static['tau'])

                for m in ("on_sonoff", "off_sonoff"):
                    syn_dynamic = states['synapses']['Dynamic'][m]
                    old_dv = syn_dynamic['dV_gSYN'][:,:,:,-1]
                    direct_source = conductance_voltage_source(
                        cur_static,
                        states['synapses']['Static'][m],
                        syn_dynamic['PSC_s'][:,:,:,-1],
                        cur_dynamic['V'][:,:,:,-1],
                    )
                    syn_dynamic['dV_gSYN'][:,:,:,-2] = old_dv
                    syn_dynamic['dV_gSYN'][:,:,:,-1] = advance_voltage_sensitivity(
                        old_dv, sonoff_dV_dV, direct_source, args['simulation']['dt']
                    )



        cur_dynamic['V_slope'] = update_holder
        cur_dynamic['V'][:,:,:,-2] = cur_dynamic['V'][:,:,:,-1]
        cur_dynamic['V'][:,:,:,-1] = cur_dynamic['V'][:,:,:,-1] + update_holder*args['simulation']['dt']
        
        #Udpate adaptation
        cur_dynamic['g_ad'][:,:,:,-2] = cur_dynamic['g_ad'][:,:,:,-1]
        cur_dynamic['g_ad'][:,:,:,-1] = cur_dynamic['g_ad'][:,:,:,-1] + (-cur_dynamic['g_ad'][:,:,:,-1]/cur_static['tau_ad'])*args['simulation']['dt']

    #Run Synapse ODES
    for k in list(states['synapses']['Static']):

        cur_syn_dynamic = states['synapses']['Dynamic'][k]
        cur_syn_static = states['synapses']['Static'][k]

        cur_syn_dynamic['PSC_s'][:,:,:,-2] = cur_syn_dynamic['PSC_s'][:,:,:,-1]
        cur_syn_dynamic['PSC_x'][:,:,:,-2] = cur_syn_dynamic['PSC_x'][:,:,:,-1]
        cur_syn_dynamic['PSC_F'][:,:,:,-2] = cur_syn_dynamic['PSC_F'][:,:,:,-1]
        cur_syn_dynamic['PSC_P'][:,:,:,-2] = cur_syn_dynamic['PSC_P'][:,:,:,-1]
        cur_syn_dynamic['PSC_q'][:,:,:,-2] = cur_syn_dynamic['PSC_q'][:,:,:,-1]

        cur_syn_dynamic['PSC_s'][:,:,:,-1] = cur_syn_dynamic['PSC_s'][:,:,:,-1] + args['simulation']['dt']*(cur_syn_static['scale']*cur_syn_dynamic['PSC_x'][:,:,:,-1] - cur_syn_dynamic['PSC_s'][:,:,:,-1])/cur_syn_static['tauR']
        cur_syn_dynamic['PSC_x'][:,:,:,-1] = cur_syn_dynamic['PSC_x'][:,:,:,-1] + args['simulation']['dt']*-cur_syn_dynamic['PSC_x'][:,:,:,-1]/cur_syn_static['tauD']
        cur_syn_dynamic['PSC_F'][:,:,:,-1] = cur_syn_dynamic['PSC_F'][:,:,:,-1] + args['simulation']['dt']*(1 - cur_syn_dynamic['PSC_F'][:,:,:,-1])/cur_syn_static['tauF']
        cur_syn_dynamic['PSC_P'][:,:,:,-1] = cur_syn_dynamic['PSC_P'][:,:,:,-1] + args['simulation']['dt']*(1 - cur_syn_dynamic['PSC_P'][:,:,:,-1])/cur_syn_static['tauP']
        cur_syn_dynamic['PSC_q'][:,:,:,-1] = cur_syn_dynamic['PSC_q'][:,:,:,-1] + args['simulation']['dt']*0
        sensitivity_names = ["gain", "alpha"]
        if k == "sonoff_ron":
            sensitivity_names.extend(("on_sonoff_gSYN", "off_sonoff_gSYN"))
        for param_name in sensitivity_names:
            if f"dPSC_x_{param_name}" in cur_syn_dynamic:
                old_dPSC_F = cur_syn_dynamic[f"dPSC_F_{param_name}"][:,:,:,-1]
                old_dPSC_P = cur_syn_dynamic[f"dPSC_P_{param_name}"][:,:,:,-1]
                old_dPSC_s = cur_syn_dynamic[f"dPSC_s_{param_name}"][:,:,:,-1]
                old_dPSC_x = cur_syn_dynamic[f"dPSC_x_{param_name}"][:,:,:,-1]
                cur_syn_dynamic[f"dPSC_F_{param_name}"][:,:,:,-2] = old_dPSC_F
                cur_syn_dynamic[f"dPSC_P_{param_name}"][:,:,:,-2] = old_dPSC_P
                cur_syn_dynamic[f"dPSC_s_{param_name}"][:,:,:,-2] = old_dPSC_s
                cur_syn_dynamic[f"dPSC_x_{param_name}"][:,:,:,-2] = old_dPSC_x
                (
                    cur_syn_dynamic[f"dPSC_F_{param_name}"][:,:,:,-1],
                    cur_syn_dynamic[f"dPSC_P_{param_name}"][:,:,:,-1],
                    cur_syn_dynamic[f"dPSC_x_{param_name}"][:,:,:,-1],
                    cur_syn_dynamic[f"dPSC_s_{param_name}"][:,:,:,-1],
                ) = advance_stp_sensitivities(
                    old_dPSC_F,
                    old_dPSC_P,
                    old_dPSC_x,
                    old_dPSC_s,
                    cur_syn_static,
                    args['simulation']['dt'],
                )

    return states

def caluclate_projections(states,k):

    projections = []

    for m in list(states['synapses']['Static'].keys()):
        if m.split('_',-1)[-1] == k:
            projections.append(m)

    return projections
