import torch

def run_conditionals(args, states,timestep):
    
    states = condtion1(args, states,timestep) #Previously condition 2b   (Absolute refractory period)
    states = condtion2(args, states,timestep) #Previously condtion 1 and 2a     (Register Spike) (Reset Voltage and adaptation)
    states = condtion3(args, states,timestep) #Previously condtion 3     (Update PSCs)
    
    return states


def condtion1(args, states,timestep):
    for k in list(states['neurons']['Static']):
        if states['neurons']['Static'][k]['output'] == 1:
            mask = torch.any((timestep <= (states['neurons']['Dynamic'][k]['tspike'] + states['neurons']['Learnable'][k]['abs_ref'][:,None,:,None]/args['simulation']['dt'])),axis=-1)
        else:
            mask = torch.any((timestep <= (states['neurons']['Dynamic'][k]['tspike'] + states['neurons']['Static'][k]['t_ref']/args['simulation']['dt'])),axis=-1)

        for m in states['neurons']['Static'][k]['projections']:
            states['synapses']['Dynamic'][m]['et'][:,:,:,-1] = states['synapses']['Dynamic'][m]['epsilon_v'][:,:,:,-1]*(~mask)
        
        if  torch.any(mask).item():
            spikers = torch.where(mask)
            states['neurons']['Dynamic'][k]['V'][spikers + (-2,)] = states['neurons']['Dynamic'][k]['V'][spikers + (-1,)] 
            states['neurons']['Dynamic'][k]['V'][spikers + (-1,)] = states['neurons']['Static'][k]['V_reset']

        if k == 'on':
            states['neurons']["Dynamic"]['strf_gain_on']["et"][:,:,:,-1] = states['neurons']["Dynamic"]['strf_gain_on']["epsilon_v"][:,:,:,-1]*(~mask)
            states['neurons']["Dynamic"]['strf_alpha_on']["et"][:,:,:,-1] = states['neurons']["Dynamic"]['strf_alpha_on']["epsilon_v"][:,:,:,-1]*(~mask)

        if k == 'off':
            states['neurons']["Dynamic"]['strf_gain_off']["et"][:,:,:,-1] = states['neurons']["Dynamic"]['strf_gain_off']["epsilon_v"][:,:,:,-1]*(~mask)
            states['neurons']["Dynamic"]['strf_alpha_off']["et"][:,:,:,-1] = states['neurons']["Dynamic"]['strf_alpha_off']["epsilon_v"][:,:,:,-1]*(~mask)

        if k == 'ron':
            states['neurons']["Dynamic"]['adaptation_ron']["et"][:,:,:,-1] = states['neurons']["Dynamic"]['adaptation_ron']["epsilon_v"][:,:,:,-1]*(~mask)
    
    return states

def condtion2(args, states,timestep):
    for k in list(states['neurons']['Static']):
        if states['neurons']['Static'][k]['output'] == 1: 
            last_spike = states['neurons']['Dynamic'][k]["tspike"].max(dim=-1).values
            p = (states['neurons']['Learnable']['ron']['rel_ref_c'][:,None,:]*torch.tanh(states['neurons']['Learnable']['ron']['rel_ref_a'][:,None,:]*(timestep-last_spike) - states['neurons']['Learnable']['ron']['rel_ref_b'][:,None,:]) + states['neurons']['Learnable']['ron']['rel_ref_c'][:,None,:])/2
            mask = ((states['neurons']['Dynamic'][k]['V'][:,:,:,-1] >= states['neurons']['Static'][k]['V_thresh']) & (torch.rand_like(p) < p)).to(torch.int64)
            states['neurons']['Dynamic'][k]["spikes_holder"][:,:,:,timestep] = mask

            for m in states['neurons']['Static'][k]['projections']:
                states['synapses']['Dynamic'][m]['et'][:,:,:,-1] = states['synapses']['Dynamic'][m]['et'][:,:,:,-1]*(p)
            #For adaptation and then eventauall probably the refractory parameters
            states['neurons']["Dynamic"]['adaptation_ron']["et"][:,:,:,-1] = states['neurons']["Dynamic"]['adaptation_ron']["et"][:,:,:,-1]*(p)
        else:
            mask = ((states['neurons']['Dynamic'][k]['V'][:,:,:,-1] >= states['neurons']['Static'][k]['V_thresh'])).to(torch.int64)

        for m in states['neurons']['Static'][k]['projections']:
            states['synapses']['Dynamic'][m]['epsilon_a'][:,:,:,-1] += states['neurons']['Dynamic'][k]['Jav'][:,:,:,-1]*states['synapses']['Dynamic'][m]['et'][:,:,:,-1]

        if torch.any(mask).item():
            spikers = torch.where(mask)
            states['neurons']['Dynamic'][k]['tspike'][spikers + (states['neurons']['Dynamic'][k]['buffer_index'][spikers].to(torch.int64)-1,)] = timestep
            states['neurons']['Dynamic'][k]['buffer_index'][spikers] = states['neurons']['Dynamic'][k]['buffer_index'][spikers] % 5 + 1
            states['neurons']['Dynamic'][k]['V'][spikers + (-2,)] = states['neurons']['Dynamic'][k]['V'][spikers + (-1,)]
            states['neurons']['Dynamic'][k]['V'][spikers + (-1,)] = states['neurons']['Static'][k]['V_reset']
            states['neurons']['Dynamic'][k]['g_ad'][spikers + (-2,)] = states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)]
            if states['neurons']['Static'][k]['output'] == 1:
                states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)] = states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)] + states['neurons']['Learnable'][k]["output_ad"][spikers[0],None,spikers[2]].T
            else:
                states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)] = states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)] + states['neurons']['Static'][k]['g_inc']

    states['neurons']["Dynamic"]['strf_gain_on']["epsilon_a"][:,:,:,-1] += states['neurons']['Dynamic']['on']['Jav'][:,:,:,-1]*states['neurons']["Dynamic"]['strf_gain_on']["et"][:,:,:,-1]
    states['neurons']["Dynamic"]['strf_alpha_on']["epsilon_a"][:,:,:,-1] += states['neurons']['Dynamic']['on']['Jav'][:,:,:,-1]*states['neurons']["Dynamic"]['strf_alpha_on']["et"][:,:,:,-1]
    states['neurons']["Dynamic"]['strf_gain_off']["epsilon_a"][:,:,:,-1] += states['neurons']['Dynamic']['off']['Jav'][:,:,:,-1]*states['neurons']["Dynamic"]['strf_gain_off']["et"][:,:,:,-1]
    states['neurons']["Dynamic"]['strf_alpha_off']["epsilon_a"][:,:,:,-1] += states['neurons']['Dynamic']['off']['Jav'][:,:,:,-1]*states['neurons']["Dynamic"]['strf_alpha_off']["et"][:,:,:,-1]
    
    states['neurons']["Dynamic"]['adaptation_ron']["epsilon_a"][:,:,:,-1] += states['neurons']['Dynamic']['ron']['Jav'][:,:,:,-1]*states['neurons']["Dynamic"]['adaptation_ron']["et"][:,:,:,-1] + states['neurons']['Dynamic']['ron']['spikes_holder'][:,:,:,timestep] #The 1 (the spike mask) is dat+1/dadaptation. (The direct path derivative)

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