import torch

def run_odes(args, states, pre_processed_spike_object, rate_object,timestep):

    #Calculate Psis
    for k in list(states['neurons']['Static']):
        w = 5
        states['neurons']['Static'][k]['psi'] = (1 - torch.tanh((states['neurons']['Dynamic'][k]['V'][:,:,:,-1] - states['neurons']['Static'][k]['V_thresh'])/w)**2)/(2*w)


    #Save last spike for refractoriness eligibilities
    ron_static = states['neurons']['Static']['ron']
    ron_dynamic = states['neurons']['Dynamic']['ron']
    last_spike = ron_dynamic["tspike"].max(dim=-1).values
        
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
            update_holder = shared_update_V - (cur_static['R']*cur_static['g_postIC']*pre_processed_spike_object['onset_offset_spks'][spks_string][:,:,:,timestep]*(cur_dynamic['V'][:,:,:,-1]-cur_static['E_exc'])) / cur_static['tau']

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


    #Eligibility calculations

    #Conductance accumulator updates
    for k in list(states['synapses']['Static']):
        post_neuron_static = states['neurons']['Static'][k.split("_")[1]]
        post_neuron_dynamic = states['neurons']['Dynamic'][k.split("_")[1]]
        local_eligiability = -post_neuron_static['R'] * states['synapses']['Dynamic'][k]['PSC_s'][:,:,:,-2]*(post_neuron_dynamic['V'][:,:,:,-2]-states['synapses']['Static'][k]['ESYN'])/post_neuron_static['tau']

        states['synapses']['Learnable'][k]['gSYN_accum'] = states['synapses']['Learnable'][k]['gSYN_accum'] + torch.sum(args['simulation']['dt']*post_neuron_static['psi']*local_eligiability,axis=1)

    #STRF related Parameters
    onset_static = states['neurons']['Static']['on']
    onset_dynamic = states['neurons']['Dynamic']['on']
    offset_static = states['neurons']['Static']['off']
    offset_dynamic = states['neurons']['Dynamic']['off']

    local_eligibility_gain_onset = -onset_static['R']*onset_static["g_postIC"]*rate_object['onset_rate_gain_deriv'][timestep,:,None,:]*(onset_dynamic['V'][:,:,:,-2]-onset_static['E_exc'])/onset_static['tau']
    local_eligibility_gain_offset = -offset_static['R']*offset_static["g_postIC"]*rate_object['offset_rate_gain_deriv'][timestep,:,None,:]*(offset_dynamic['V'][:,:,:,-2]-offset_static['E_exc'])/offset_static['tau']

    local_eligibility_alpha_onset = -onset_static['R']*onset_static["g_postIC"]*rate_object['onset_rate_deriv'][timestep,:,None,:]*(onset_dynamic['V'][:,:,:,-2]-onset_static['E_exc'])/onset_static['tau']
    local_eligibility_alpha_offset = -offset_static['R']*offset_static["g_postIC"]*rate_object['offset_rate_deriv'][timestep,:,None,:]*(offset_dynamic['V'][:,:,:,-2]-offset_static['E_exc'])/offset_static['tau']

    states['neurons']['Learnable']['STRF_gain_accum'] = states['neurons']['Learnable']['STRF_gain_accum'] +  torch.sum((onset_static['psi']*(args['simulation']['dt']/1000)*args['simulation']['dt']*local_eligibility_gain_onset) +
                                                                                                                           (offset_static['psi']*(args['simulation']['dt']/1000)*args['simulation']['dt']*local_eligibility_gain_offset),axis=1)
    states['neurons']['Learnable']['STRF_alpha_accum'] = states['neurons']['Learnable']['STRF_alpha_accum'] + torch.sum((onset_static['psi']*(args['simulation']['dt']/1000)*args['simulation']['dt']*local_eligibility_alpha_onset) +
                                                                                                                           (offset_static['psi']*(args['simulation']['dt']/1000)*args['simulation']['dt']*local_eligibility_alpha_offset),axis=1)

    local_eligiability_ad = ((ron_dynamic['V'][:,:,:,-2]-ron_static['E_k'])/ron_static['tau'])
    states['neurons']['Learnable']['ron']["output_ad_accum"] = states['neurons']['Learnable']['ron']["output_ad_accum"] + torch.sum(-ron_static['R']*ron_static['psi']*local_eligiability_ad,axis=1)  

    local_abs = (ron_dynamic['V'][:,:,:,-2] - ron_static['V_reset'])*(-(1-(torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:]/args['simulation']['dt'])**2)))
    local_a = torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:])*states["neurons"]["Learnable"]["ron"]["rel_ref_c"][:,None,:]*(timestep-last_spike)*(1-torch.tanh(states["neurons"]["Learnable"]["ron"]["rel_ref_a"][:,None,:]*(timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["rel_ref_b"][:,None,:])**2)
    local_b = torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:])*states["neurons"]["Learnable"]["ron"]["rel_ref_c"][:,None,:]*(-1)*(1-torch.tanh(states["neurons"]["Learnable"]["ron"]["rel_ref_a"][:,None,:]*(timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["rel_ref_b"][:,None,:])**2)
    local_c = torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:]/args['simulation']['dt']) * (torch.tanh(states["neurons"]["Learnable"]["ron"]["rel_ref_a"][:,None,:]*(timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["rel_ref_b"][:,None,:]) + 1)

    states["neurons"]["Learnable"]["ron"]["abs_ref_accum"] = states["neurons"]["Learnable"]["ron"]["abs_ref_accum"] + torch.sum(ron_static['psi']*local_abs,axis=1)
    states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum"] = states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum"] + torch.sum(ron_static['psi']*local_a,axis=1)
    states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum"] = states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum"] + torch.sum(ron_static['psi']*local_b,axis=1)
    states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum"] = states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum"] + torch.sum(ron_static['psi']*local_c,axis=1)

    #Update CV grads here
    states["neurons"]["Learnable"]["ron"]["abs_ref_accum_cv"] = states["neurons"]["Learnable"]["ron"]["abs_ref_accum_cv"] + local_abs
    states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum_cv"] = states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum_cv"] + local_a
    states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum_cv"] = states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum_cv"] + local_b
    states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum_cv"] = states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum_cv"] + local_c

    #Store Bk for now
    states['neurons']['Learnable']['Bk'] = -states['synapses']['Learnable']['sonoff_ron']['gSYN']*(states['neurons']['Static']['ron']['V_thresh']-states['synapses']['Static']['sonoff_ron']['ESYN'])

    return states

def caluclate_projections(states,k):

    projections = []

    for m in list(states['synapses']['Static'].keys()):
        if m.split('_',-1)[-1] == k:
            projections.append(m)

    return projections