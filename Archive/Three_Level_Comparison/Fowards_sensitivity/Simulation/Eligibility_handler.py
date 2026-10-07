import torch

def update_eligibility(args, states, rate_object, timestep):

    #Calculate conducatance eligibilities
    for k in list(states['synapses']['Static']):
        post_neuron_static = states['neurons']['Static'][k.split("_")[1]]
        post_neuron_dynamic = states['neurons']['Dynamic'][k.split("_")[1]]
        cur_syn_running = states['synapses']['Running_grads'][k]
        cur_syn_dynamic = states['synapses']['Dynamic'][k]
        cur_syn_static = states['synapses']['Static'][k]

        #Update sensitivities for dz/dg quantitity in psc updates for gsyns
        if k == "on_sonoff" or k == "off_sonoff":
            cur_syn_running["du_dg_circular"][:,:,:,:-1] = cur_syn_running["du_dg_circular"][:,:,:,1:].clone()
            cur_syn_running["du_dg_circular"][:,:,:,-1] = states['neurons']['Static'][k.split("_")[-1]]['abs_indicator']*states['neurons']['Static'][k.split("_")[-1]]['psi']*states['synapses']["Running_grads"][k]["total_running_contribution"][:,:,:,-2]

        if k == "sonoff_ron":

            for m in ["on","off"]:
                
                du_dg = states['synapses']["Running_grads"][f'{m}_sonoff']["du_dg_circular"][:,:,:,0]

                cur_syn_running[f'dPSCx{m}'][:,:,:,-2] = cur_syn_running[f'dPSCx{m}'][:,:,:,-1]
                cur_syn_running[f'dPSCF{m}'][:,:,:,-2] = cur_syn_running[f'dPSCF{m}'][:,:,:,-1]
                cur_syn_running[f'dPSCP{m}'][:,:,:,-2] = cur_syn_running[f'dPSCP{m}'][:,:,:,-1]

                cur_syn_running[f'dPSCx{m}'][:,:,:,-1] = cur_syn_running[f'dPSCx{m}'][:,:,:,-1] + du_dg*cur_syn_dynamic['PSC_F'][:,:,:,-2]*cur_syn_dynamic['PSC_P'][:,:,:,-2]
                cur_syn_running[f'dPSCF{m}'][:,:,:,-1] = cur_syn_running[f'dPSCF{m}'][:,:,:,-1] + du_dg*cur_syn_static['PSC_fF']*(cur_syn_static['PSC_maxF']-cur_syn_dynamic['PSC_F'][:,:,:,-2])
                cur_syn_running[f'dPSCP{m}'][:,:,:,-1] = cur_syn_running[f'dPSCP{m}'][:,:,:,-1] + du_dg*cur_syn_dynamic['PSC_P'][:,:,:,-2]*(-cur_syn_static['PSC_fP'])

    
        if k.split("_")[1] == "ron":
            states['synapses']['Learnable'][k]['gSYN_accum'] = states['synapses']['Learnable'][k]['gSYN_accum'] + torch.sum(states['neurons']['Static'][k.split("_")[-1]]['psi']*states['synapses']["Running_grads"][k]["total_running_contribution"][:,:,:,-2]*states['neurons']['Static'][k.split("_")[-1]]['probability']*states['neurons']['Static'][k.split("_")[-1]]['abs_indicator'],axis=1)
        else:
            if k.split("_")[0] == "on":
                states['synapses']['Learnable'][k]['gSYN_accum'] = states['synapses']['Learnable'][k]['gSYN_accum'] + torch.sum(states['neurons']['Static']["ron"]['probability']*states['neurons']['Static']["ron"]['abs_indicator']*states['neurons']['Static']["ron"]['psi']*states['synapses']['Running_grads']['sonoff_ron']["hidden_running_contribution_on"][:,:,:,-2],axis=1)
            else:
                states['synapses']['Learnable'][k]['gSYN_accum'] = states['synapses']['Learnable'][k]['gSYN_accum'] + torch.sum(states['neurons']['Static']["ron"]['probability']*states['neurons']['Static']["ron"]['abs_indicator']*states['neurons']['Static']["ron"]['psi']*states['synapses']['Running_grads']['sonoff_ron']["hidden_running_contribution_off"][:,:,:,-2],axis=1)


        #STRF PSC updates
        for m in ['gain','alpha']:
            cur_syn_running[f"du_dg_circular_{m}"][:,:,:,:-1] = cur_syn_running[f"du_dg_circular_{m}"][:,:,:,1:].clone()

            if k.split("_")[0] != "sonoff":
                cur_syn_running[f"du_dg_circular_{m}"][:,:,:,-1] = states['neurons']['Static'][k.split("_")[0]]['abs_indicator']*states['neurons']['Static'][k.split("_")[0]]['psi']*states['neurons']["Running_grads"][k.split("_")[0]][f"total_running_contribution_{m}"][:,:,:,-2]
            else:
                cur_syn_running[f"du_dg_circular_{m}"][:,:,:,-1] = states['neurons']['Static'][k.split("_")[0]]['abs_indicator']*states['neurons']['Static'][k.split("_")[0]]['psi']*(states['synapses']["Running_grads"]['on_sonoff'][f"total_running_contribution_{m}"][:,:,:,-2]+states['synapses']["Running_grads"]['off_sonoff'][f"total_running_contribution_{m}"][:,:,:,-2])

            du_dg = states['synapses']["Running_grads"][k][f"du_dg_circular_{m}"][:,:,:,0]

            # Update the PSC values
            cur_syn_running[f'dPSCx{m}'][:,:,:,-2] = cur_syn_running[f'dPSCx{m}'][:,:,:,-1]
            cur_syn_running[f'dPSCF{m}'][:,:,:,-2] = cur_syn_running[f'dPSCF{m}'][:,:,:,-1]
            cur_syn_running[f'dPSCP{m}'][:,:,:,-2] = cur_syn_running[f'dPSCP{m}'][:,:,:,-1]

            cur_syn_running[f'dPSCx{m}'][:,:,:,-1] = cur_syn_running[f'dPSCx{m}'][:,:,:,-1] + du_dg*cur_syn_dynamic['PSC_F'][:,:,:,-2]*cur_syn_dynamic['PSC_P'][:,:,:,-2]
            cur_syn_running[f'dPSCF{m}'][:,:,:,-1] = cur_syn_running[f'dPSCF{m}'][:,:,:,-1] + du_dg*cur_syn_static['PSC_fF']*(cur_syn_static['PSC_maxF']-cur_syn_dynamic['PSC_F'][:,:,:,-2])
            cur_syn_running[f'dPSCP{m}'][:,:,:,-1] = cur_syn_running[f'dPSCP{m}'][:,:,:,-1] + du_dg*cur_syn_dynamic['PSC_P'][:,:,:,-2]*(-cur_syn_static['PSC_fP'])

    for m in ['gain','alpha']:

        full_path = states['neurons']["Static"]["ron"]['psi']*states['neurons']["Static"]["ron"]['abs_indicator']*states['neurons']['Static']["ron"]['probability']*(states['synapses']["Running_grads"]["sonoff_ron"][f"total_running_contribution_{m}"][:,:,:,-2] + states['synapses']["Running_grads"]["off_ron"][f"total_running_contribution_{m}"][:,:,:,-2] + states['synapses']["Running_grads"]["on_ron"][f"total_running_contribution_{m}"][:,:,:,-2])

        if m == "gain":
            states['neurons']['Learnable']['STRF_gain_accum'] = states['neurons']['Learnable']['STRF_gain_accum'] + torch.sum(full_path,axis=1)
        else:
            states['neurons']['Learnable']['STRF_alpha_accum'] = states['neurons']['Learnable']['STRF_alpha_accum'] + torch.sum(full_path,axis=1)

    
    #Calculate adaptation eligibilities
    ron_static = states['neurons']['Static']['ron']
    ron_dynamic = states['neurons']['Dynamic']['ron']
    #states['neurons']['Learnable']['ron']["output_ad_accum"] = states['neurons']['Learnable']['ron']["output_ad_accum"] + torch.sum(-ron_static['R']*ron_static['psi']*(((1+torch.tanh(ron_dynamic['V'][:,:,:,-1]-ron_static['V_thresh']))/2)*((1+torch.tanh(-(ron_dynamic['V'][:,:,:,-2]-ron_static['V_thresh'])))/2) * args['simulation']['dt'] * (ron_dynamic['V'][:,:,:,-1]-ron_static['E_k'])/ron_static['tau']),axis=1)  
    states['neurons']['Learnable']['ron']["output_ad_accum"] = states['neurons']['Learnable']['ron']["output_ad_accum"] + torch.sum(states['neurons']['Static']["ron"]['probability']*states['neurons']['Static']["ron"]['abs_indicator']*states['neurons']['Static']["ron"]['psi']*states['neurons']['Running_grads']['ron']["output_ad_contribution"][:,:,:,-2],axis=1)  
    
    #Calculate refractoriness eligibilities
    last_spike = ron_dynamic["tspike"].max(dim=-1).values
    # Absolute refractory compilation.
    # states["neurons"]["Learnable"]["ron"]["abs_ref_accum"] = states["neurons"]["Learnable"]["ron"]["abs_ref_accum"] + torch.sum(states['neurons']['Static']["ron"]['probability']*states["neurons"]["Running_grads"]['ron']["abs_ref_voltage"][:,:,:,-2]*states['neurons']['Static']["ron"]['psi'],axis=1)

    states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum"] = states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum"] + torch.sum(ron_static['psi']*0.5*(1+torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:]/args['simulation']['dt']))*states["neurons"]["Learnable"]["ron"]["rel_ref_c"][:,None,:]*(timestep-last_spike)*(1-torch.tanh(states["neurons"]["Learnable"]["ron"]["rel_ref_a"][:,None,:]*(timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["rel_ref_b"][:,None,:])**2),axis=1)
    states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum"] = states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum"] + torch.sum(ron_static['psi']*0.5*(1+torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:]/args['simulation']['dt']))*states["neurons"]["Learnable"]["ron"]["rel_ref_c"][:,None,:]*(-1)*(1-torch.tanh(states["neurons"]["Learnable"]["ron"]["rel_ref_a"][:,None,:]*(timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["rel_ref_b"][:,None,:])**2),axis=1)
    states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum"] = states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum"] + torch.sum(ron_static['psi']*0.5*(1+torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:]/args['simulation']['dt']))*(torch.tanh(states["neurons"]["Learnable"]["ron"]["rel_ref_a"][:,None,:]*(timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["rel_ref_b"][:,None,:])) + ron_static['psi']*0.5*(1+torch.tanh((timestep-last_spike)-states["neurons"]["Learnable"]["ron"]["abs_ref"][:,None,:]/args['simulation']['dt'])),axis=1)
    
    

    return states
