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

    states['neurons']['Learnable']['STRF_gain_accum'] = states['neurons']['Learnable']['STRF_gain_accum'] + ((torch.sum(onset_static['psi']*-onset_static['R']*onset_static['g_postIC']*30 * (onset_dynamic['V'][:,:,:,-1]-onset_static['E_exc']),axis=1) * (1 - torch.tanh((args['simulation']['dt']/1000)*rate_object['onset_rate'][timestep,:,:]-1.5)**2)/onset_static['tau']) + 
                                                                                                            (torch.sum(offset_static['psi']*-offset_static['R']*offset_static['g_postIC']*30  * (offset_dynamic['V'][:,:,:,-1]-offset_static['E_exc']),axis=1)* (1 - torch.tanh((args['simulation']['dt']/1000)*rate_object['offset_rate'][timestep,:,:]-1.5)**2)/offset_static['tau']))

    states['neurons']['Learnable']['STRF_alpha_accum'] = states['neurons']['Learnable']['STRF_alpha_accum'] + ((torch.sum(onset_static['psi']*-onset_static['R']*onset_static['g_postIC']*30 * (onset_dynamic['V'][:,:,:,-1]-onset_static['E_exc']),axis=1)* (1 - torch.tanh((args['simulation']['dt']/1000)*rate_object['onset_rate_deriv'][timestep,:,:]-1.5)**2)/onset_static['tau']) +
                                                                                                            (torch.sum(offset_static['psi']*-offset_static['R']*offset_static['g_postIC']*30 * (offset_dynamic['V'][:,:,:,-1]-offset_static['E_exc']),axis=1)* (1 - torch.tanh((args['simulation']['dt']/1000)*rate_object['offset_rate_deriv'][timestep,:,:]-1.5)**2) /offset_static['tau']))
    
    #Calculate adaptation eligibilities
    ron_static = states['neurons']['Static']['ron']
    ron_dynamic = states['neurons']['Dynamic']['ron']
    states['neurons']['Learnable']['ron']["output_ad_accum"] = states['neurons']['Learnable']['ron']["output_ad_accum"] + torch.sum(-ron_static['R']*ron_static['psi']*(((1+torch.tanh(ron_dynamic['V'][:,:,:,-1]-ron_static['V_thresh']))/2)*((1+torch.tanh(-(ron_dynamic['V'][:,:,:,-2]-ron_static['V_thresh'])))/2) * args['simulation']['dt'] * (ron_dynamic['V'][:,:,:,-1]-ron_static['E_k'])/ron_static['tau']),axis=1)  
    
    #Calculate refractoriness eligibilities
    last_spike = ron_dynamic["tspike"].max(dim=-1).values
    states["neurons"]["Learnable"]["ron"]["abs_ref_accum"] = states["neurons"]["Learnable"]["ron"]["abs_ref_accum"] + torch.sum(ron_static['psi']*(ron_dynamic['V'][:,:,:,-1] - ron_static['V_reset'])*(-(1-(torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:])**2))),axis=1)
    states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum"] = states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum"] + torch.sum(ron_static['psi']*torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:])*states["neurons"]["Learnable"]["ron"]["rel_ref_c"][:,None,:]*(timestep-last_spike)*(1-torch.tanh(states["neurons"]["Learnable"]["ron"]["rel_ref_a"][:,None,:]*(timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["rel_ref_b"][:,None,:])**2),axis=1)
    states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum"] = states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum"] + torch.sum(ron_static['psi']*torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:])*states["neurons"]["Learnable"]["ron"]["rel_ref_c"][:,None,:]*(-1)*(1-torch.tanh(states["neurons"]["Learnable"]["ron"]["rel_ref_a"][:,None,:]*(timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["rel_ref_b"][:,None,:])**2),axis=1)
    states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum"] = states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum"] + torch.sum(ron_static['psi']*torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:])*(torch.tanh(states["neurons"]["Learnable"]["ron"]["rel_ref_a"][:,None,:]*(timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["rel_ref_b"][:,None,:])) + ron_static['psi']*torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:]),axis=1)
    
    #Store Bk for now
    states['neurons']['Learnable']['Bk'] = -states['synapses']['Learnable']['sonoff_ron']['gSYN']*(states['neurons']['Static']['ron']['V_thresh']-states['synapses']['Static']['sonoff_ron']['ESYN'])

    return states