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
    strf_input_scale = args['simulation']['dt'] * (args['simulation']['dt']/1000)
    
    paths = {}

    for k in list(states['synapses']['Static']):
        post_neuron_static = states['neurons']['Static'][k.split("_")[1]]
        post_neuron_dynamic = states['neurons']['Dynamic'][k.split("_")[1]]
        pre_neuron_dynamic = states['neurons']['Dynamic'][k.split("_")[0]]
        cur_syn_dynamic = states['synapses']['Dynamic'][k]
        cur_syn_static = states['synapses']['Static'][k]
        last_spike = pre_neuron_dynamic["tspike"].max(dim=-1).values
        paths[f'{k}'] = -post_neuron_static['R']*post_neuron_static['psi']*cur_syn_dynamic['PSC_s'][:,:,:,-1]*(1+2*(1-torch.tanh(timestep-last_spike)**2))*states['synapses']['Learnable'][k]['gSYN'][:,None,:]*(post_neuron_dynamic['V'][:,:,:,-1]-cur_syn_static['ESYN'])/post_neuron_static['tau']
    
    onset_paths = paths['on_ron']
    offset_paths = paths['off_ron']

    #Store Bk for now
    states['neurons']['Learnable']['Bk'] = -states['synapses']['Learnable']['sonoff_ron']['gSYN']*(states['neurons']['Static']['ron']['V_thresh']-states['synapses']['Static']['sonoff_ron']['ESYN'])


    onset_strf_voltage_factor = torch.sum(onset_paths*states['neurons']['Static']['on']['psi']*-onset_static['R']*onset_static['g_postIC']*(onset_dynamic['V'][:,:,:,-1]-onset_static['E_exc']),axis=1)
    offset_strf_voltage_factor = torch.sum(offset_paths*states['neurons']['Static']['off']['psi']*-offset_static['R']*offset_static['g_postIC']*(offset_dynamic['V'][:,:,:,-1]-offset_static['E_exc']),axis=1)

    


    states['neurons']['Learnable']['STRF_gain_accum'] = states['neurons']['Learnable']['STRF_gain_accum'] + ((onset_strf_voltage_factor * strf_input_scale*rate_object['onset_rate_gain_deriv'][timestep,:,:])/onset_static['tau'] +
                                                                                                            (offset_strf_voltage_factor * strf_input_scale*rate_object['offset_rate_gain_deriv'][timestep,:,:])/offset_static['tau'])

    states['neurons']['Learnable']['STRF_alpha_accum'] = states['neurons']['Learnable']['STRF_alpha_accum'] + ((onset_strf_voltage_factor * strf_input_scale*rate_object['onset_rate_deriv'][timestep,:,:])/onset_static['tau'] +
                                                                                                            (offset_strf_voltage_factor * strf_input_scale*rate_object['offset_rate_deriv'][timestep,:,:])/offset_static['tau'])
    
    #Calculate adaptation eligibilities
    ron_static = states['neurons']['Static']['ron']
    ron_dynamic = states['neurons']['Dynamic']['ron']
    states['neurons']['Learnable']['ron']["output_ad_accum"] = states['neurons']['Learnable']['ron']["output_ad_accum"] + torch.sum(-ron_static['R']*ron_static['psi']*(((1+torch.tanh(ron_dynamic['V'][:,:,:,-1]-ron_static['V_thresh']))/2)*((1+torch.tanh(-(ron_dynamic['V'][:,:,:,-2]-ron_static['V_thresh'])))/2) * args['simulation']['dt'] * (ron_dynamic['V'][:,:,:,-1]-ron_static['E_k'])/ron_static['tau']),axis=1)  
    
    #Calculate refractoriness eligibilities
    last_spike = ron_dynamic["tspike"].max(dim=-1).values
    states["neurons"]["Learnable"]["ron"]["abs_ref_accum"] = states["neurons"]["Learnable"]["ron"]["abs_ref_accum"] + torch.sum(ron_static['psi']*(ron_dynamic['V'][:,:,:,-1] - ron_static['V_reset'])*(-0.5/args['simulation']['dt']*(1-(torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:]/args['simulation']['dt'])**2))),axis=1)
    states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum"] = states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum"] + torch.sum(ron_static['psi']*0.5*(1+torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:]/args['simulation']['dt']))*states["neurons"]["Learnable"]["ron"]["rel_ref_c"][:,None,:]*(timestep-last_spike)*(1-torch.tanh(states["neurons"]["Learnable"]["ron"]["rel_ref_a"][:,None,:]*(timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["rel_ref_b"][:,None,:])**2),axis=1)
    states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum"] = states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum"] + torch.sum(ron_static['psi']*0.5*(1+torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:]/args['simulation']['dt']))*states["neurons"]["Learnable"]["ron"]["rel_ref_c"][:,None,:]*(-1)*(1-torch.tanh(states["neurons"]["Learnable"]["ron"]["rel_ref_a"][:,None,:]*(timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["rel_ref_b"][:,None,:])**2),axis=1)
    states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum"] = states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum"] + torch.sum(ron_static['psi']*0.5*(1+torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:]/args['simulation']['dt']))*(torch.tanh(states["neurons"]["Learnable"]["ron"]["rel_ref_a"][:,None,:]*(timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["rel_ref_b"][:,None,:])) + ron_static['psi']*0.5*(1+torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:]/args['simulation']['dt'])),axis=1)
    
    

    return states
