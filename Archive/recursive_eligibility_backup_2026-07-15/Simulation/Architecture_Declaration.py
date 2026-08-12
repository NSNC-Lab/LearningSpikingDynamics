import torch

def build_network(args, device, params):
    
    
    neuron_init = {"Static": {}, "Dynamic": {}, "Learnable": {}}
    synapase_init = {"Static": {}, "Dynamic": {}, "Learnable": {}}

    neuron_init = declare_neuron_properties(args, device, params, neuron_init, name = "on",input=1)
    neuron_init = declare_neuron_properties(args, device, params, neuron_init, name = "off",input=1)
    neuron_init = declare_neuron_properties(args, device, params, neuron_init, name = "sonoff", g_inc=0, E_L=-57,V_reset=-52,t_ref=0.5,g_L=1/100)
    neuron_init = declare_neuron_properties(args, device, params, neuron_init, name = "ron", noise=1, output=1)

    synapase_init = declare_synapse_properties(args, device, params, synapase_init, name = "on_ron")
    synapase_init = declare_synapse_properties(args, device, params, synapase_init, name = "off_ron")
    synapase_init = declare_synapse_properties(args, device, params, synapase_init, name = "on_sonoff",  PSC_fP=0.2,tauP = 80,  tauR = 0.1, tauD = 1, PSC_delay = 3)
    synapase_init = declare_synapse_properties(args, device, params, synapase_init, name = "off_sonoff", PSC_fP=0,  tauP = 80,  tauR = 0.1, tauD = 1, PSC_delay = 3)
    synapase_init = declare_synapse_properties(args, device, params, synapase_init, name = "sonoff_ron", PSC_fP=0.5,tauP = 120, tauR = 1,   tauD = 4.5, PSC_delay = 0.5, ESYN=-80)  

    #Declare Learnable params
    neuron_init["Learnable"]["STRF_gain"] = torch.nn.Parameter(torch.tensor((params['Strf_gain']), device=device, dtype=torch.float32)) 
    neuron_init["Learnable"]["STRF_alpha"] = torch.nn.Parameter(torch.tensor((params['Strf_alpha']), device=device, dtype=torch.float32)) 

    neuron_init["Learnable"]["STRF_gain_accum"] = torch.zeros((params['Strf_gain'].shape), device=device, dtype=torch.float32)
    neuron_init["Learnable"]["STRF_gain_grad"] = torch.zeros((params['Strf_gain'].shape), device=device, dtype=torch.float32) 
    
    neuron_init["Learnable"]["STRF_alpha_accum"] = torch.zeros((params['Strf_alpha'].shape), device=device, dtype=torch.float32)
    neuron_init["Learnable"]["STRF_alpha_grad"] = torch.zeros((params['Strf_alpha'].shape), device=device, dtype=torch.float32)  

    neuron_init["Learnable"]["STRF_gain_tracker"] = torch.zeros((*params['Strf_gain'].shape,args['simulation']['epochs']), device=device, dtype=torch.float32)
    neuron_init["Learnable"]["STRF_alpha_tracker"] = torch.zeros((*params['Strf_alpha'].shape,args['simulation']['epochs']), device=device, dtype=torch.float32)

    #Adam Parameters
    neuron_init["Adam"] = {}
    neuron_init["Adam"]["t"] = 0
    neuron_init["Adam"]["m"] = torch.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets']),args['simulation']['num_params']), device=device, dtype=torch.float32)
    neuron_init["Adam"]["v"] = torch.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets']),args['simulation']['num_params']), device=device, dtype=torch.float32)

    neuron_init["BookKeeping"] = []

    return {"neurons": neuron_init, "synapses": synapase_init}

def declare_neuron_properties(args, device, params, this_neuron_init, name, C = 0.1, g_L = 1/200, E_L = -65, t_ref = 0, E_k = -80, tau_ad = 100, g_inc = 0.0003, Itonic = 0 , V_thresh = -47, V_reset = -54, g_postIC = 0.17, E_exc = 0, nSYN = 0.015, noise_E_exc = 0, tauR_N = 0.7, tauD_N = 1.5, noise = 0,output=0, input=0):

    #Static

    this_neuron_init["Static"][name] = {}
    this_neuron_init["Static"][name]["E_L"] = E_L
    this_neuron_init["Static"][name]["t_ref"] = t_ref
    this_neuron_init["Static"][name]["E_k"] = E_k
    this_neuron_init["Static"][name]["tau_ad"] = tau_ad
    this_neuron_init["Static"][name]["Itonic"] = Itonic
    this_neuron_init["Static"][name]["V_thresh"] = V_thresh
    this_neuron_init["Static"][name]["V_reset"] = V_reset
    this_neuron_init["Static"][name]["g_postIC"] = g_postIC
    this_neuron_init["Static"][name]["E_exc"] = E_exc

    this_neuron_init["Static"][name]["R"] = 1/g_L
    this_neuron_init["Static"][name]["tau"] = C*this_neuron_init["Static"][name]["R"]

    this_neuron_init["Static"][name]["noise"] = noise
    this_neuron_init["Static"][name]["input"] = input
    this_neuron_init["Static"][name]["output"] = output

    #Dyanmic

    this_neuron_init["Dynamic"][name] = {}
    this_neuron_init["Dynamic"][name]["V"] = torch.ones((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32) * E_L
    this_neuron_init["Dynamic"][name]["g_ad"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
    this_neuron_init["Dynamic"][name]["tspike"] = torch.ones((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),5), device=device, dtype=torch.float32) * -30
    this_neuron_init["Dynamic"][name]["buffer_index"] = torch.ones((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets'])), device=device, dtype=torch.int64)
    # Current-step event flag.  Delayed synapses enqueue this mask directly;
    # the short tspike ring above is retained only for last-spike/refractory
    # bookkeeping and must not be used as a pending-event queue.
    this_neuron_init["Dynamic"][name]["spike_mask"] = torch.zeros(
        (args['simulation']['batch_size'], 10, len(args['simulation']['cell_targets'])),
        device=device,
        dtype=torch.bool,
    )

    if input == 1:
        # Persistent input-neuron voltage sensitivities.  They are advanced
        # after the forward Euler step once the STRF rate derivatives for the
        # current timestep are available.
        for param_name in ("gain", "alpha"):
            this_neuron_init["Dynamic"][name][f"dV_STRF_{param_name}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        this_neuron_init["Dynamic"][name]["input_V_jacobian"] = torch.full(
            (args['simulation']['batch_size'],10,len(args['simulation']['cell_targets'])),
            -1.0 / this_neuron_init["Static"][name]["tau"],
            device=device,
            dtype=torch.float32,
        )

    #Noise specific
    if noise == 1:
        this_neuron_init["Static"][name]["nSYN"] = nSYN
        this_neuron_init["Static"][name]["noise_E_exc"] = noise_E_exc
        this_neuron_init["Static"][name]["tauR_N"] = tauR_N
        this_neuron_init["Static"][name]["tauD_N"] = tauD_N
        this_neuron_init["Static"][name]["noise_scale"] = (tauD_N/tauR_N)**(tauR_N/(tauD_N-tauR_N))
        this_neuron_init["Dynamic"][name]["noise_sn"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        this_neuron_init["Dynamic"][name]["noise_xn"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)

    #Output Specific
    if output == 1:
        this_neuron_init["Dynamic"][name]["spikes_holder"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),args['simulation']['sim_len']), device=device, dtype=torch.int64)
        this_neuron_init["Dynamic"][name]["mean_sse_loss"] = 0
        this_neuron_init["Dynamic"][name]["mean_CV_loss"] = 0
        # Persistent state sensitivities for the output-adaptation increment.
        # These are separate from the per-PSTH-bin eligibility accumulator.
        this_neuron_init["Dynamic"][name]["dV_output_ad"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        this_neuron_init["Dynamic"][name]["dg_ad_output_ad"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        
        # Persistent state sensitivities for the output-refractoriness increment.
        this_neuron_init["Dynamic"][name]["dV_abs_ref"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        this_neuron_init["Dynamic"][name]["dV_rel_ref_a"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        this_neuron_init["Dynamic"][name]["dV_rel_ref_b"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)   
        this_neuron_init["Dynamic"][name]["dV_rel_ref_c"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)

    #Learnable
    if output == 1:
        #Adapatation
        this_neuron_init["Learnable"][name] = {}
        this_neuron_init["Learnable"][name]["output_ad"] = torch.nn.Parameter(torch.tensor((params['output_ad']), device=device, dtype=torch.float32)) 
        this_neuron_init["Learnable"][name]["output_ad_accum"] = torch.zeros((params['output_ad'].shape), device=device, dtype=torch.float32)
        this_neuron_init["Learnable"][name]["output_ad_grad"] = torch.zeros((params['output_ad'].shape), device=device, dtype=torch.float32)
        this_neuron_init["Learnable"][name]["output_ad_tracker"] = torch.zeros((*params['output_ad'].shape,args['simulation']['epochs']), device=device, dtype=torch.float32)

        #Refractory period
        this_neuron_init['Learnable'][name]['abs_ref'] = torch.nn.Parameter(torch.tensor((params['abs_ref']), device=device, dtype=torch.float32))
        this_neuron_init['Learnable'][name]['abs_ref_accum'] = torch.zeros((params['abs_ref'].shape), device=device, dtype=torch.float32)
        this_neuron_init['Learnable'][name]['abs_ref_grad'] = torch.zeros((params['abs_ref'].shape), device=device, dtype=torch.float32)
        this_neuron_init['Learnable'][name]['abs_ref_tracker'] = torch.zeros((*params['abs_ref'].shape,args['simulation']['epochs']), device=device, dtype=torch.float32)
        this_neuron_init['Learnable'][name]['rel_ref_a'] = torch.nn.Parameter(torch.tensor((params['rel_ref_a']), device=device, dtype=torch.float32))
        this_neuron_init['Learnable'][name]['rel_ref_a_accum'] = torch.zeros((params['rel_ref_a'].shape), device=device, dtype=torch.float32)
        this_neuron_init['Learnable'][name]['rel_ref_a_grad'] = torch.zeros((params['rel_ref_a'].shape), device=device, dtype=torch.float32)
        this_neuron_init['Learnable'][name]['rel_ref_a_tracker'] = torch.zeros((*params['rel_ref_a'].shape,args['simulation']['epochs']), device=device, dtype=torch.float32)
        this_neuron_init['Learnable'][name]['rel_ref_b'] = torch.nn.Parameter(torch.tensor((params['rel_ref_b']), device=device, dtype=torch.float32)) 
        this_neuron_init['Learnable'][name]['rel_ref_b_accum'] = torch.zeros((params['rel_ref_b'].shape), device=device, dtype=torch.float32)
        this_neuron_init['Learnable'][name]['rel_ref_b_grad'] = torch.zeros((params['rel_ref_b'].shape), device=device, dtype=torch.float32)
        this_neuron_init['Learnable'][name]['rel_ref_b_tracker'] = torch.zeros((*params['rel_ref_b'].shape,args['simulation']['epochs']), device=device, dtype=torch.float32)
        this_neuron_init['Learnable'][name]['rel_ref_c'] = torch.nn.Parameter(torch.tensor((params['rel_ref_c']), device=device, dtype=torch.float32))
        this_neuron_init['Learnable'][name]['rel_ref_c_accum'] = torch.zeros((params['rel_ref_c'].shape), device=device, dtype=torch.float32)
        this_neuron_init['Learnable'][name]['rel_ref_c_grad'] = torch.zeros((params['rel_ref_c'].shape), device=device, dtype=torch.float32)
        this_neuron_init['Learnable'][name]['rel_ref_c_tracker'] = torch.zeros((*params['rel_ref_c'].shape,args['simulation']['epochs']), device=device, dtype=torch.float32)

    else:
        this_neuron_init["Static"][name]["g_inc"] = g_inc

    return this_neuron_init

def declare_synapse_properties(args, device, params, this_synapse_init, name, ESYN = 0, tauD = 1.5, tauR = 0.7, PSC_delay = 1, gSYN = 0.001, PSC_fF = 0, PSC_fP = 0.1 , tauF = 180 , tauP = 30 , PSC_maxF = 4):

    #Static

    this_synapse_init["Static"][name] = {}
    this_synapse_init["Static"][name]["ESYN"] = ESYN
    this_synapse_init["Static"][name]["tauD"] = tauD
    this_synapse_init["Static"][name]["tauR"] = tauR
    this_synapse_init["Static"][name]["PSC_delay"] = PSC_delay
    this_synapse_init["Static"][name]["gSYN"] = gSYN
    this_synapse_init["Static"][name]["PSC_fF"] = PSC_fF
    this_synapse_init["Static"][name]["PSC_fP"] = PSC_fP 
    this_synapse_init["Static"][name]["tauF"] = tauF 
    this_synapse_init["Static"][name]["tauP"] = tauP 
    this_synapse_init["Static"][name]["PSC_maxF"] = PSC_maxF 
    this_synapse_init["Static"][name]["scale"] = (tauD/tauR)**(tauR/(tauD-tauR)) 

    #Dyanmic
    this_synapse_init["Dynamic"][name] = {}
    this_synapse_init["Dynamic"][name]["PSC_s"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
    this_synapse_init["Dynamic"][name]["PSC_x"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
    this_synapse_init["Dynamic"][name]["PSC_F"] = torch.ones((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
    this_synapse_init["Dynamic"][name]["PSC_P"] = torch.ones((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
    this_synapse_init["Dynamic"][name]["PSC_q"] = torch.ones((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
    delay_steps = int(PSC_delay/args['simulation']['dt'])
    # One slot per possible presynaptic timestep through the static delay.
    # This is lossless even when more than five spikes are simultaneously in
    # flight, unlike reconstructing arrivals from the five-entry tspike ring.
    this_synapse_init["Dynamic"][name]["spike_delay_queue"] = torch.zeros(
        (
            args['simulation']['batch_size'],
            10,
            len(args['simulation']['cell_targets']),
            delay_steps + 1,
        ),
        device=device,
        dtype=torch.bool,
    )

    if name in ("on_ron", "off_ron", "on_sonoff", "off_sonoff", "sonoff_ron"):
        # Voltage sensitivity to this synapse's own conductance.  Unlike
        # gSYN_accum, this trace persists across PSTH bins.
        this_synapse_init["Dynamic"][name]["dV_gSYN"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        for param_name in ("gain", "alpha"):
            # Persistent short-term-plasticity state tangents.  F is currently
            # constant because every PSC_fF is zero, but retaining dF keeps the
            # event map complete and makes future facilitation changes safe.
            this_synapse_init["Dynamic"][name][f"dPSC_F_{param_name}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
            this_synapse_init["Dynamic"][name][f"dPSC_P_{param_name}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
            this_synapse_init["Dynamic"][name][f"dPSC_x_{param_name}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
            this_synapse_init["Dynamic"][name][f"dPSC_s_{param_name}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
            this_synapse_init["Dynamic"][name][f"dV_{param_name}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
            if name == "sonoff_ron":
                # Raw continuous arrival-time derivative, in milliseconds per
                # parameter.  It is deliberately not divided by tauD.
                this_synapse_init["Dynamic"][name][f"dtime_{param_name}_delay"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),delay_steps+1), device=device, dtype=torch.float32)
            else:
                # Fixed-grid straight-through derivative of the hard on/off
                # event strength.  This has different units and a different
                # event map from dtime_*.
                this_synapse_init["Dynamic"][name][f"dsoft_{param_name}_delay"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),delay_steps+1), device=device, dtype=torch.float32)

        if name == "sonoff_ron":
            # Sensitivities that carry each upstream sonoff conductance through
            # sonoff spike timing, the inhibitory PSC, and finally ron voltage.
            for source_name in ("on_sonoff", "off_sonoff"):
                trace_name = f"{source_name}_gSYN"
                this_synapse_init["Dynamic"][name][f"dPSC_F_{trace_name}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
                this_synapse_init["Dynamic"][name][f"dPSC_P_{trace_name}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
                this_synapse_init["Dynamic"][name][f"dPSC_x_{trace_name}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
                this_synapse_init["Dynamic"][name][f"dPSC_s_{trace_name}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
                this_synapse_init["Dynamic"][name][f"dV_{trace_name}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
                this_synapse_init["Dynamic"][name][f"dtime_{trace_name}_delay"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),delay_steps+1), device=device, dtype=torch.float32)

    #Learnable
    this_synapse_init["Learnable"][name] = {}
    this_synapse_init["Learnable"][name]["gSYN"] = torch.nn.Parameter(torch.tensor((params[f"{name}_gSYN"]), device=device, dtype=torch.float32))
    this_synapse_init["Learnable"][name]["gSYN_accum"] = torch.zeros((params[f"{name}_gSYN"].shape), device=device, dtype=torch.float32)
    this_synapse_init["Learnable"][name]["gSYN_grad"] = torch.zeros((params[f"{name}_gSYN"].shape), device=device, dtype=torch.float32)
    this_synapse_init["Learnable"][name]["gSYN_tracker"] = torch.zeros((*params[f"{name}_gSYN"].shape,args['simulation']['epochs']), device=device, dtype=torch.float32)

    return this_synapse_init

