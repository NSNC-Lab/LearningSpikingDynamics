import torch

def update_eligibility(args, states, rate_object, timestep):
    
    #Calculate Psis
    for k in list(states['neurons']['Static']):
        states['neurons']['Static'][k]['psi'] = (1 - torch.tanh(states['neurons']['Dynamic'][k]['V'][:,:,:,-1] - states['neurons']['Static'][k]['V_thresh'])**2)

    #Calculate conducatance eligibilities
    for k in list(states['synapses']['Static']):
        post_neuron_static = states['neurons']['Static'][k.split("_")[1]]
        post_neuron_dynamic = states['neurons']['Dynamic'][k.split("_")[1]]
        states['synapses']['Learnable'][k]['gSYN_accum'] = states['synapses']['Learnable'][k]['gSYN_accum'] + torch.sum(args['simulation']['dt']*post_neuron_static['psi']*-post_neuron_static['R'] * states['synapses']['Dynamic'][k]['PSC_s'][:,:,:,-1]*(post_neuron_dynamic['V'][:,:,:,-1]-states['synapses']['Static'][k]['ESYN'])/post_neuron_static['tau'],axis=1)

    #Calculate strf eligibilities
    onset_static = states['neurons']['Static']['on']
    onset_dynamic = states['neurons']['Dynamic']['on']
    offset_static = states['neurons']['Static']['off']
    offset_dynamic = states['neurons']['Dynamic']['off']
    ron_static = states['neurons']['Static']['ron']
    sononoff_static = states['neurons']['Static']['sonoff']
    sononoff_dynamic = states['neurons']['Dynamic']['sonoff']
    strf_input_scale = args['simulation']['dt'] * (args['simulation']['dt']/1000)

    #Store Bk for now
    states['neurons']['Learnable']['Bk'] = -states['synapses']['Learnable']['sonoff_ron']['gSYN']*(states['neurons']['Static']['ron']['V_thresh']-states['synapses']['Static']['sonoff_ron']['ESYN'])

    states['neurons']['Learnable']['STRF_gain_accum'] = states['neurons']['Learnable']['STRF_gain_accum'] + torch.sum(ron_static['psi']*(states['synapses']['Dynamic']['on_ron']['dV_gain'][:,:,:,-1] + states['synapses']['Dynamic']['off_ron']['dV_gain'][:,:,:,-1] + states['synapses']['Dynamic']['sonoff_ron']['dV_gain'][:,:,:,-1]),axis=1)
    states['neurons']['Learnable']['STRF_alpha_accum'] = states['neurons']['Learnable']['STRF_alpha_accum'] + torch.sum(ron_static['psi']*(states['synapses']['Dynamic']['on_ron']['dV_alpha'][:,:,:,-1] + states['synapses']['Dynamic']['off_ron']['dV_alpha'][:,:,:,-1] + states['synapses']['Dynamic']['sonoff_ron']['dV_alpha'][:,:,:,-1]),axis=1)

    local_pre_derivs = {
        "on_ron": {
            "gain": onset_static['psi']*-onset_static['R']*onset_static['g_postIC']*(onset_dynamic['V'][:,:,:,-2]-onset_static['E_exc'])*strf_input_scale*rate_object['onset_rate_gain_deriv'][timestep,:,None,:]/onset_static['tau'],
            "alpha": onset_static['psi']*-onset_static['R']*onset_static['g_postIC']*(onset_dynamic['V'][:,:,:,-2]-onset_static['E_exc'])*strf_input_scale*rate_object['onset_rate_deriv'][timestep,:,None,:]/onset_static['tau'],
        },
        "off_ron": {
            "gain": offset_static['psi']*-offset_static['R']*offset_static['g_postIC']*(offset_dynamic['V'][:,:,:,-2]-offset_static['E_exc'])*strf_input_scale*rate_object['offset_rate_gain_deriv'][timestep,:,None,:]/offset_static['tau'],
            "alpha": offset_static['psi']*-offset_static['R']*offset_static['g_postIC']*(offset_dynamic['V'][:,:,:,-2]-offset_static['E_exc'])*strf_input_scale*rate_object['offset_rate_deriv'][timestep,:,None,:]/offset_static['tau'],
        },
        "on_sonoff": {
            "gain": onset_static['psi']*-onset_static['R']*onset_static['g_postIC']*(onset_dynamic['V'][:,:,:,-2]-onset_static['E_exc'])*strf_input_scale*rate_object['onset_rate_gain_deriv'][timestep,:,None,:]/onset_static['tau'],
            "alpha": onset_static['psi']*-onset_static['R']*onset_static['g_postIC']*(onset_dynamic['V'][:,:,:,-2]-onset_static['E_exc'])*strf_input_scale*rate_object['onset_rate_deriv'][timestep,:,None,:]/onset_static['tau'],
        },
        "off_sonoff": {
            "gain": offset_static['psi']*-offset_static['R']*offset_static['g_postIC']*(offset_dynamic['V'][:,:,:,-2]-offset_static['E_exc'])*strf_input_scale*rate_object['offset_rate_gain_deriv'][timestep,:,None,:]/offset_static['tau'],
            "alpha": offset_static['psi']*-offset_static['R']*offset_static['g_postIC']*(offset_dynamic['V'][:,:,:,-2]-offset_static['E_exc'])*strf_input_scale*rate_object['offset_rate_deriv'][timestep,:,None,:]/offset_static['tau'],
        },
        "sonoff_ron": {
            "gain": sononoff_static['psi']*(states['synapses']['Dynamic']['on_sonoff']['dV_gain'][:,:,:,-1] + states['synapses']['Dynamic']['off_sonoff']['dV_gain'][:,:,:,-1]),
            "alpha": sononoff_static['psi']*(states['synapses']['Dynamic']['on_sonoff']['dV_alpha'][:,:,:,-1] + states['synapses']['Dynamic']['off_sonoff']['dV_alpha'][:,:,:,-1]),
        }
    }
    for syn_name in ("on_ron", "off_ron", "sonoff_ron", "on_sonoff", "off_sonoff"):
        delay_steps = int(states['synapses']['Static'][syn_name]['PSC_delay']/args['simulation']['dt'])
        delay_idx = timestep % (delay_steps + 1)
        read_idx = (timestep - delay_steps) % (delay_steps + 1)
        for param_name in ("gain", "alpha"):
            syn_dynamic = states['synapses']['Dynamic'][syn_name]
            syn_dynamic[f"dpre_{param_name}_delay"][:,:,:,delay_idx] = local_pre_derivs[syn_name][param_name]
            if timestep >= delay_steps:
                syn_dynamic[f"dPSC_x_{param_name}"][:,:,:,-1] = syn_dynamic[f"dPSC_x_{param_name}"][:,:,:,-1] + syn_dynamic['PSC_q'][:,:,:,-1]*syn_dynamic[f"dpre_{param_name}_delay"][:,:,:,read_idx]
    
    #Calculate adaptation eligibilities
    ron_dynamic = states['neurons']['Dynamic']['ron']
    states['neurons']['Learnable']['ron']["output_ad_accum"] = states['neurons']['Learnable']['ron']["output_ad_accum"] + torch.sum(-ron_static['R']*ron_static['psi']*(((1+torch.tanh(ron_dynamic['V'][:,:,:,-1]-ron_static['V_thresh']))/2)*((1+torch.tanh(-(ron_dynamic['V'][:,:,:,-2]-ron_static['V_thresh'])))/2) * args['simulation']['dt'] * (ron_dynamic['V'][:,:,:,-1]-ron_static['E_k'])/ron_static['tau']),axis=1)  
    
    #Calculate refractoriness eligibilities
    last_spike = ron_dynamic["tspike"].max(dim=-1).values
    states["neurons"]["Learnable"]["ron"]["abs_ref_accum"] = states["neurons"]["Learnable"]["ron"]["abs_ref_accum"] + torch.sum(ron_static['psi']*(ron_dynamic['V'][:,:,:,-1] - ron_static['V_reset'])*(-0.5/args['simulation']['dt']*(1-(torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:]/args['simulation']['dt'])**2))),axis=1)
    states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum"] = states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum"] + torch.sum(ron_static['psi']*0.5*(1+torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:]/args['simulation']['dt']))*states["neurons"]["Learnable"]["ron"]["rel_ref_c"][:,None,:]*(timestep-last_spike)*(1-torch.tanh(states["neurons"]["Learnable"]["ron"]["rel_ref_a"][:,None,:]*(timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["rel_ref_b"][:,None,:])**2),axis=1)
    states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum"] = states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum"] + torch.sum(ron_static['psi']*0.5*(1+torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:]/args['simulation']['dt']))*states["neurons"]["Learnable"]["ron"]["rel_ref_c"][:,None,:]*(-1)*(1-torch.tanh(states["neurons"]["Learnable"]["ron"]["rel_ref_a"][:,None,:]*(timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["rel_ref_b"][:,None,:])**2),axis=1)
    states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum"] = states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum"] + torch.sum(ron_static['psi']*0.5*(1+torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:]/args['simulation']['dt']))*(torch.tanh(states["neurons"]["Learnable"]["ron"]["rel_ref_a"][:,None,:]*(timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["rel_ref_b"][:,None,:])) + ron_static['psi']*0.5*(1+torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:]/args['simulation']['dt'])),axis=1)
    
    

    return states
