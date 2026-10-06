import torch

def run_conditionals(args, states,timestep,rate_object):
    
    states = condtion1(args, states,timestep) #Previously condition 2b   (Absolute refractory period)
    states = condtion2(args, states,timestep,rate_object) #Previously condtion 1 and 2a     (Register Spike) (Reset Voltage and adaptation)
    states = condtion3(args, states,timestep) #Previously condtion 3     (Update PSCs)
    
    return states


def condtion1(args, states,timestep):
    for k in list(states['neurons']['Static']):
        if states['neurons']['Static'][k]['output'] == 1:
            mask = torch.any((timestep <= (states['neurons']['Dynamic'][k]['tspike'] + states['neurons']['Learnable'][k]['abs_ref'][:,None,:,None]/args['simulation']['dt'])),axis=-1)
        else:
            mask = torch.any((timestep <= (states['neurons']['Dynamic'][k]['tspike'] + states['neurons']['Static'][k]['t_ref']/args['simulation']['dt'])),axis=-1)
        if  torch.any(mask).item():
            spikers = torch.where(mask)
            states['neurons']['Dynamic'][k]['V'][spikers + (-2,)] = states['neurons']['Dynamic'][k]['V'][spikers + (-1,)] 
            states['neurons']['Dynamic'][k]['V'][spikers + (-1,)] = states['neurons']['Static'][k]['V_reset']
    return states

def condtion2(args, states,timestep,rate_object):
    for k in list(states['neurons']['Static']):
        if states['neurons']['Static'][k]['output'] == 1:
            last_spike = states['neurons']['Dynamic'][k]["tspike"].max(dim=-1).values
            mask = ((states['neurons']['Dynamic'][k]['V'][:,:,:,-1] >= states['neurons']['Static'][k]['V_thresh']) & (torch.rand(last_spike.shape,device=torch.device(args['simulation']['device']))<states['neurons']['Learnable']['ron']['rel_ref_c'][:,None,:]*torch.tanh(states['neurons']['Learnable']['ron']['rel_ref_a'][:,None,:]*(timestep-last_spike) - states['neurons']['Learnable']['ron']['rel_ref_b'][:,None,:]) + states['neurons']['Learnable']['ron']['rel_ref_c'][:,None,:])).to(torch.int64)
            states['neurons']['Dynamic'][k]["spikes_holder"][:,:,:,timestep] = mask
        else:
            mask = ((states['neurons']['Dynamic'][k]['V'][:,:,:,-1] >= states['neurons']['Static'][k]['V_thresh'])).to(torch.int64)
        if torch.any(mask).item():

            spikers = torch.where(mask)

            write_slot = states['neurons']['Dynamic'][k]['buffer_index'][spikers] - 1
            latest_slot = (write_slot - 1) % 5
            previous_slot = (write_slot - 2) % 5

            latest_time = states['neurons']['Dynamic'][k]["tspike"][spikers + (latest_slot,)]
            previous_time = states['neurons']['Dynamic'][k]["tspike"][spikers + (previous_slot,)]

            new_interval = timestep - latest_time
            previous_interval = latest_time - previous_time
            Itot = torch.where(previous_time >= 0, previous_interval - new_interval, torch.zeros_like(new_interval))

            #Conductance accumulator updates
            for z in list(states['synapses']['Static']):

                if k == z.split("_")[1]:

                    post_neuron_static = states['neurons']['Static'][z.split("_")[1]]
                    post_neuron_dynamic = states['neurons']['Dynamic'][z.split("_")[1]]
                    local_eligiability = -post_neuron_static['R'] * states['synapses']['Dynamic'][z]['PSC_s'][spikers + (-2,)]*(post_neuron_dynamic['V'][spikers + (-2,)]-states['synapses']['Static'][z]['ESYN'])/post_neuron_static['tau']

                    states['synapses']['Learnable'][z]['gSYN_accum_cv'][spikers] = states['synapses']['Learnable'][z]['gSYN_accum_cv'][spikers] + states['synapses']['Learnable'][z]['gSYN_accum_cv_storage'][spikers]*Itot
                    states['synapses']['Learnable'][z]['gSYN_accum_cv_storage'][spikers] = -args['simulation']['dt']*local_eligiability/(post_neuron_dynamic['V'][spikers + (-1,)]-post_neuron_dynamic['V'][spikers + (-2,)])

            if k == 'off':
                offset_static = states['neurons']['Static']['off']
                offset_dynamic = states['neurons']['Dynamic']['off']
                local_eligibility_gain_offset = -(args['simulation']['dt']/1000)*offset_static['R']*offset_static["g_postIC"]*rate_object['offset_rate_gain_deriv'][timestep,spikers[0],spikers[2]]*(offset_dynamic['V'][spikers + (-2,)]-offset_static['E_exc'])/offset_static['tau']
                local_eligibility_alpha_offset = -(args['simulation']['dt']/1000)*offset_static['R']*offset_static["g_postIC"]*rate_object['offset_rate_deriv'][timestep,spikers[0],spikers[2]]*(offset_dynamic['V'][spikers + (-2,)]-offset_static['E_exc'])/offset_static['tau']

                states['neurons']['Learnable']['STRF_gain_accum_cv'][spikers] = states['neurons']['Learnable']['STRF_gain_accum_cv'][spikers] + (states['neurons']['Learnable']['STRF_gain_accum_cv_storage_off'][spikers])*Itot
                states['neurons']['Learnable']['STRF_alpha_accum_cv'][spikers] = states['neurons']['Learnable']['STRF_alpha_accum_cv'][spikers] + (states['neurons']['Learnable']['STRF_alpha_accum_cv_storage_off'][spikers])*Itot

                states['neurons']['Learnable']['STRF_gain_accum_cv_storage_off'][spikers] = (-args['simulation']['dt']*local_eligibility_gain_offset/(offset_dynamic['V'][spikers + (-1,)]-offset_dynamic['V'][spikers + (-2,)])).float()
                states['neurons']['Learnable']['STRF_alpha_accum_cv_storage_off'][spikers] = (-args['simulation']['dt']*local_eligibility_alpha_offset/(offset_dynamic['V'][spikers + (-1,)]-offset_dynamic['V'][spikers + (-2,)])).float()

            if k == 'on':
                onset_static = states['neurons']['Static']['on']
                onset_dynamic = states['neurons']['Dynamic']['on']
                local_eligibility_gain_onset = -(args['simulation']['dt']/1000)*onset_static['R']*onset_static["g_postIC"]*rate_object['onset_rate_gain_deriv'][timestep,spikers[0],spikers[2]]*(onset_dynamic['V'][spikers + (-2,)]-onset_static['E_exc'])/onset_static['tau']
                local_eligibility_alpha_onset = (args['simulation']['dt']/1000)*-onset_static['R']*onset_static["g_postIC"]*rate_object['onset_rate_deriv'][timestep,spikers[0],spikers[2]]*(onset_dynamic['V'][spikers + (-2,)]-onset_static['E_exc'])/onset_static['tau']

                states['neurons']['Learnable']['STRF_gain_accum_cv'][spikers] = states['neurons']['Learnable']['STRF_gain_accum_cv'][spikers] + (states['neurons']['Learnable']['STRF_gain_accum_cv_storage_on'][spikers])*Itot
                states['neurons']['Learnable']['STRF_alpha_accum_cv'][spikers] = states['neurons']['Learnable']['STRF_alpha_accum_cv'][spikers] + (states['neurons']['Learnable']['STRF_alpha_accum_cv_storage_on'][spikers])*Itot

                states['neurons']['Learnable']['STRF_gain_accum_cv_storage_on'][spikers] = (-args['simulation']['dt']*local_eligibility_gain_onset/(onset_dynamic['V'][spikers + (-1,)]-onset_dynamic['V'][spikers + (-2,)])).float() 
                states['neurons']['Learnable']['STRF_alpha_accum_cv_storage_on'][spikers] = (-args['simulation']['dt']*local_eligibility_alpha_onset/(onset_dynamic['V'][spikers + (-1,)]-onset_dynamic['V'][spikers + (-2,)])).float()

            #states['neurons']['Learnable']['STRF_gain_accum_cv'] = states['neurons']['Learnable']['STRF_gain_accum_cv'] + -args['simulation']['dt']*local_eligibility_gain_onset/(post_neuron_dynamic['V'][spikers + (-1,)]-post_neuron_dynamic['V'][spikers + (-2,)]) -args['simulation']['dt']*local_eligibility_gain_offset/(post_neuron_dynamic['V'][spikers + (-1,)]-post_neuron_dynamic['V'][spikers + (-2,)])

            #states['neurons']['Learnable']['STRF_alpha_accum_cv'] = states['neurons']['Learnable']['STRF_alpha_accum_cv'] + -args['simulation']['dt']*local_eligibility_alpha_onset/(post_neuron_dynamic['V'][spikers + (-1,)]-post_neuron_dynamic['V'][spikers + (-2,)]) -args['simulation']['dt']*local_eligibility_alpha_offset/(post_neuron_dynamic['V'][spikers + (-1,)]-post_neuron_dynamic['V'][spikers + (-2,)])

            if k == 'ron':
                ron_static = states['neurons']['Static']['ron']
                ron_dynamic = states['neurons']['Dynamic']['ron']

                r_before = ron_dynamic['g_ad'][spikers + (-2,)] / states['neurons']['Learnable']['ron']['output_ad'][spikers[0], spikers[2]]
                local_eligiability_ad = -ron_static['R']*((ron_dynamic['V'][spikers + (-2,)]-ron_static['E_k'])/ron_static['tau'])
                states['neurons']['Learnable']['ron']["output_ad_accum_cv"][spikers] = states['neurons']['Learnable']['ron']["output_ad_accum_cv"][spikers] + states['neurons']['Learnable']['ron']["output_ad_accum_cv_storage"][spikers]*Itot
                states['neurons']['Learnable']['ron']["output_ad_accum_cv_storage"][spikers] = -args['simulation']['dt'] * r_before * local_eligiability_ad / (ron_dynamic['V'][spikers + (-1,)] - ron_dynamic['V'][spikers + (-2,)])

                # last_selected = last_spike[spikers]
                
                # local_abs = (ron_dynamic['V'][spikers + (-2,)] - ron_static['V_reset'])*(-(1-(torch.tanh((timestep-last_selected)-states["neurons"]["Learnable"]["ron"]["abs_ref"][spikers[0],spikers[2]]/args['simulation']['dt'])**2)))
                # local_a = torch.tanh((timestep-last_selected)-states["neurons"]["Learnable"]["ron"]["abs_ref"][spikers[0],spikers[2]]/args['simulation']['dt'])*states["neurons"]["Learnable"]["ron"]["rel_ref_c"][spikers[0],spikers[2]]*(timestep-last_selected)*(1-torch.tanh(states["neurons"]["Learnable"]["ron"]["rel_ref_a"][spikers[0],spikers[2]]*(timestep-last_selected)-states["neurons"]["Learnable"]["ron"]["rel_ref_b"][spikers[0],spikers[2]])**2)
                # local_b = torch.tanh((timestep-last_selected)-states["neurons"]["Learnable"]["ron"]["abs_ref"][spikers[0],spikers[2]]/args['simulation']['dt'])*states["neurons"]["Learnable"]["ron"]["rel_ref_c"][spikers[0],spikers[2]]*(-1)*(1-torch.tanh(states["neurons"]["Learnable"]["ron"]["rel_ref_a"][spikers[0],spikers[2]]*(timestep-last_selected)-states["neurons"]["Learnable"]["ron"]["rel_ref_b"][spikers[0],spikers[2]])**2)
                # local_c = torch.tanh((timestep-last_selected)-states["neurons"]["Learnable"]["ron"]["abs_ref"][spikers[0],spikers[2]]/args['simulation']['dt']) * (torch.tanh(states["neurons"]["Learnable"]["ron"]["rel_ref_a"][spikers[0],spikers[2]]*(timestep-last_selected)-states["neurons"]["Learnable"]["ron"]["rel_ref_b"][spikers[0],spikers[2]]) + 1)
            
                # #Update CV grads here
                # states["neurons"]["Learnable"]["ron"]["abs_ref_accum_cv"][spikers] = states["neurons"]["Learnable"]["ron"]["abs_ref_accum_cv"][spikers] + states["neurons"]["Learnable"]["ron"]["abs_ref_accum_cv_storage"][spikers]*Itot
                # states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum_cv"][spikers] = states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum_cv"][spikers] + states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum_cv_storage"][spikers]*Itot
                # states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum_cv"][spikers] = states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum_cv"][spikers] + states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum_cv_storage"][spikers]*Itot
                # states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum_cv"][spikers] = states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum_cv"][spikers] + states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum_cv_storage"][spikers]*Itot
                # states["neurons"]["Learnable"]["ron"]["abs_ref_accum_cv_storage"][spikers] = -args['simulation']['dt']*local_abs
                # states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum_cv_storage"][spikers] = -args['simulation']['dt']*local_a
                # states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum_cv_storage"][spikers] = -args['simulation']['dt']*local_b
                # states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum_cv_storage"][spikers] = -args['simulation']['dt']*local_c

                
            states['neurons']['Dynamic'][k]['tspike'][spikers + (states['neurons']['Dynamic'][k]['buffer_index'][spikers].to(torch.int64)-1,)] = timestep
            states['neurons']['Dynamic'][k]['buffer_index'][spikers] = states['neurons']['Dynamic'][k]['buffer_index'][spikers] % 5 + 1
            states['neurons']['Dynamic'][k]['V'][spikers + (-2,)] = states['neurons']['Dynamic'][k]['V'][spikers + (-1,)]
            states['neurons']['Dynamic'][k]['V'][spikers + (-1,)] = states['neurons']['Static'][k]['V_reset']
            states['neurons']['Dynamic'][k]['g_ad'][spikers + (-2,)] = states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)]
            if states['neurons']['Static'][k]['output'] == 1:
                states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)] = states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)] + states['neurons']['Learnable'][k]["output_ad"][spikers[0],None,spikers[2]].T
            else:
                states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)] = states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)] + states['neurons']['Static'][k]['g_inc']
    
    return states

def condtion3(args, states,timestep):
    for k in list(states['synapses']['Static']):
        pre_neuron_dynamic = states['neurons']['Dynamic'][k.split("_")[0]]
        mask = torch.any((timestep == ((pre_neuron_dynamic['tspike'] + int(states['synapses']['Static'][k]['PSC_delay']/args['simulation']['dt']))).to(torch.int64)),axis=-1)
        if torch.any(mask).item():
            spikers = torch.where(mask)
            states['synapses']['Dynamic'][k]['PSC_x'][spikers + (-2,)] = states['synapses']['Dynamic'][k]['PSC_x'][spikers + (-1,)]
            states['synapses']['Dynamic'][k]['PSC_q'][spikers + (-2,)] = states['synapses']['Dynamic'][k]['PSC_q'][spikers + (-1,)]
            states['synapses']['Dynamic'][k]['PSC_F'][spikers + (-2,)] = states['synapses']['Dynamic'][k]['PSC_F'][spikers + (-1,)]
            states['synapses']['Dynamic'][k]['PSC_P'][spikers + (-2,)] = states['synapses']['Dynamic'][k]['PSC_P'][spikers + (-1,)]
            states['synapses']['Dynamic'][k]['PSC_x'][spikers + (-1,)] = states['synapses']['Dynamic'][k]['PSC_x'][spikers + (-1,)] + states['synapses']['Dynamic'][k]['PSC_q'][spikers + (-1,)]
            states['synapses']['Dynamic'][k]['PSC_q'][spikers + (-1,)] = states['synapses']['Dynamic'][k]['PSC_F'][spikers + (-1,)] * states['synapses']['Dynamic'][k]['PSC_P'][spikers + (-1,)]
            states['synapses']['Dynamic'][k]['PSC_F'][spikers + (-1,)] = states['synapses']['Dynamic'][k]['PSC_F'][spikers + (-1,)] + states['synapses']['Static'][k]['PSC_fF']*(states['synapses']['Static'][k]['PSC_maxF']-states['synapses']['Dynamic'][k]['PSC_F'][spikers + (-1,)])
            states['synapses']['Dynamic'][k]['PSC_P'][spikers + (-1,)] = states['synapses']['Dynamic'][k]['PSC_P'][spikers + (-1,)] * (1-states['synapses']['Static'][k]['PSC_fP'])
    return states