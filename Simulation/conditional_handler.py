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
        if  torch.any(mask).item():
            spikers = torch.where(mask)
            states['neurons']['Dynamic'][k]['V'][spikers + (-2,)] = states['neurons']['Dynamic'][k]['V'][spikers + (-1,)] 
            states['neurons']['Dynamic'][k]['V'][spikers + (-1,)] = states['neurons']['Static'][k]['V_reset']

            #Sensitivities
            #Reset the sensitivies related to voltage
            for m in states['neurons']['Static'][k.split("_")[-1]]['projections']:
                cur_running_grads = states['synapses']['Running_grads'][m]
                cur_running_grads['total_running_contribution'][:,:,:,-1] = cur_running_grads['total_running_contribution'][:,:,:,-1]*(1-mask*1)
                for z in ['gain','alpha']:
                    cur_running_grads[f'total_running_contribution_{z}'][:,:,:,-1] = cur_running_grads[f'total_running_contribution_{z}'][:,:,:,-1]*(1-mask*1)

            if k == "on" or k == "off":
                cur_running_grads = states['neurons']['Running_grads'][k]
                for z in ['gain','alpha']:
                    cur_running_grads[f'total_running_contribution_{z}'][:,:,:,-1] = cur_running_grads[f'total_running_contribution_{z}'][:,:,:,-1]*(1-mask*1)


            if k == "ron":
                hidden_running_grads = states['synapses']['Running_grads']['sonoff_ron']
                for m in ["on", "off"]:
                    hidden_running_grads[f'hidden_running_contribution_{m}'][:,:,:,-1] = hidden_running_grads[f'hidden_running_contribution_{m}'][:,:,:,-1]*(1-mask*1)

                states['neurons']['Running_grads']['ron']["output_ad_contribution"][:,:,:,-1] = states['neurons']['Running_grads']['ron']["output_ad_contribution"][:,:,:,-1]*(1-mask*1)

        states['neurons']['Static'][k]['abs_indicator'] = (1-mask*1) #<- Here we want to multiply by 0 only when we are in the abosulate refractory period so I invert the mask.

    return states

def condtion2(args, states,timestep):
    for k in list(states['neurons']['Static']):

        #Calculate Psis
        w = 5
        xt = (states['neurons']['Dynamic'][k]['V'][:,:,:,-1] - states['neurons']['Static'][k]['V_thresh'])/w
        qt = (1+torch.tanh(xt))/2
        states['neurons']['Static'][k]['psi'] = (1 - torch.tanh(xt)**2)/(2*w) #Moved phi calculation here so that it would be done before V_reset can be applied.
        spike_score_weights = qt
        time_offsets = torch.arange(-10,1,device=torch.device(args['simulation']['device']))
        time_tensor = (timestep + time_offsets) * args['simulation']['dt']

        if k == 'ron':
            #These buffers are a way to implement the CV loss and convert our smooth psi aligned derivatvies to spike timing derivatives that can be used for the CV derivative calculation.
            #The main idea is that a spike time can be derived by taking a bunch of spike scores (essential the surrogate values calculated in each condition) muliplying by the spike time and then
            #Averaging the spike time to get the point that most likely corresponds to a spike time. These buffers store the partials are then needed to calculate the gradient later.
            states['neurons']["Running_grads"][k]["Circular_window_buffer"][:,:,:,:-1] = states['neurons']["Running_grads"][k]["Circular_window_buffer"][:,:,:,1:].clone()
            states['neurons']["Running_grads"][k]["Circular_window_buffer"][:,:,:,-1] = spike_score_weights

            states['neurons']["Running_grads"][k]["Circular_window_buffer_voltage_sensitivity"][:,:,:,:-1] = states['neurons']["Running_grads"][k]["Circular_window_buffer_voltage_sensitivity"][:,:,:,1:].clone()
            states['neurons']["Running_grads"][k]["Circular_window_buffer_voltage_sensitivity"][:,:,:,-1] = states['neurons']['Static'][k]['psi']*states['neurons']["Running_grads"][k]["abs_ref_voltage"][:,:,:,-1]

        if states['neurons']['Static'][k]['output'] == 1:
            last_spike = states['neurons']['Dynamic'][k]["tspike"].max(dim=-1).values
            probability = torch.clamp(states['neurons']['Learnable']['ron']['rel_ref_c'][:,None,:]*torch.tanh(states['neurons']['Learnable']['ron']['rel_ref_a'][:,None,:]*(timestep-last_spike) - states['neurons']['Learnable']['ron']['rel_ref_b'][:,None,:]) + states['neurons']['Learnable']['ron']['rel_ref_c'][:,None,:],min=0.0,max=1.0)
            mask = ((states['neurons']['Dynamic'][k]['V'][:,:,:,-1] >= states['neurons']['Static'][k]['V_thresh']) & (torch.rand(last_spike.shape,device=torch.device(args['simulation']['device']))<probability)).to(torch.int64)
            states['neurons']['Dynamic'][k]["spikes_holder"][ :,:,:,timestep] = mask
            states['neurons']['Static'][k]['probability'] = probability

            #This portion finishes out the recursion for the a,b,and c portions of the realtive refacotry period

            #Surrogate for the probaiblity function
            a_t_a_prime = states['neurons']['Learnable']['ron']['rel_ref_c'][:,None,:]*(timestep-last_spike)*(1-torch.tanh(states['neurons']['Learnable']['ron']['rel_ref_a'][:,None,:]*(timestep-last_spike)-states['neurons']['Learnable']['ron']['rel_ref_b'][:,None,:])**2)
            a_t_b_prime = -states['neurons']['Learnable']['ron']['rel_ref_c'][:,None,:]*(1-torch.tanh(states['neurons']['Learnable']['ron']['rel_ref_a'][:,None,:]*(timestep-last_spike)-states['neurons']['Learnable']['ron']['rel_ref_b'][:,None,:])**2)
            a_t_c_prime = torch.tanh(states['neurons']['Learnable']['ron']['rel_ref_a'][:,None,:]*(timestep-last_spike) - states['neurons']['Learnable']['ron']['rel_ref_b'][:,None,:]) + 1

            #Voltage recursion
            voltage_a = states["neurons"]["Running_grads"]['ron']["a_ref_voltage"]
            voltage_b = states["neurons"]["Running_grads"]['ron']["b_ref_voltage"]
            voltage_c = states["neurons"]["Running_grads"]['ron']["c_ref_voltage"]
            adaptation_a = states["neurons"]["Running_grads"]['ron']["a_ref_adaptation"]
            adaptation_b = states["neurons"]["Running_grads"]['ron']["b_ref_adaptation"]
            adaptation_c = states["neurons"]["Running_grads"]['ron']["c_ref_adaptation"]

            #Populate the buffer with the "spike score" for condition 2 (relative refractoriness)
            #Note 8-7. Expanding width of surrogate for stability.
            #width = 5 #In mv
            #x_val = xt
            q_rel = qt
            psi_rel = states['neurons']['Static'][k]['psi']
            states['neurons']["Running_grads"][k]["Circular_window_buffer_rel"][:,:,:,:-1] = states['neurons']["Running_grads"][k]["Circular_window_buffer_rel"][:,:,:,1:].clone()
            states['neurons']["Running_grads"][k]["Circular_window_buffer_rel"][:,:,:,-1] = q_rel*probability*states['neurons']['Static'][k]['abs_indicator']

            #Notably, E_tot here is considering the entire mask for condition 2. The entire mask's surrogate is p*q. E_tot is the derivative of that
            E_tot_a = (q_rel*a_t_a_prime + probability*psi_rel*voltage_a[:,:,:,-2])*states['neurons']['Static'][k]['abs_indicator']
            E_tot_b = (q_rel*a_t_b_prime + probability*psi_rel*voltage_b[:,:,:,-2])*states['neurons']['Static'][k]['abs_indicator']
            E_tot_c = (q_rel*a_t_c_prime + probability*psi_rel*voltage_c[:,:,:,-2])*states['neurons']['Static'][k]['abs_indicator']
            voltage_a[:,:,:,-1] = ((1-mask)*voltage_a[:,:,:,-2] -E_tot_a*(states['neurons']['Dynamic']['ron']['V'][:,:,:,-1] - states['neurons']['Static']['ron']['V_reset']))*states['neurons']['Static'][k]['abs_indicator']
            voltage_b[:,:,:,-1] = ((1-mask)*voltage_b[:,:,:,-2] -E_tot_b*(states['neurons']['Dynamic']['ron']['V'][:,:,:,-1] - states['neurons']['Static']['ron']['V_reset']))*states['neurons']['Static'][k]['abs_indicator']
            voltage_c[:,:,:,-1] = ((1-mask)*voltage_c[:,:,:,-2] -E_tot_c*(states['neurons']['Dynamic']['ron']['V'][:,:,:,-1] - states['neurons']['Static']['ron']['V_reset']))*states['neurons']['Static'][k]['abs_indicator']
            voltage_a[:,:,:,-2] = voltage_a[:,:,:,-1]
            voltage_b[:,:,:,-2] = voltage_b[:,:,:,-1]
            voltage_c[:,:,:,-2] = voltage_c[:,:,:,-1]
            # if timestep == 28000:
            #     print('voltage_a')
            #     print(voltage_a)
            #     print('e_tot_a')
            #     print(E_tot_a)

            #Sensitivity storage for CV calculation
            states['neurons']["Running_grads"][k]["Circular_window_buffer_voltage_a_sensitivity"][:,:,:,:-1] = states['neurons']["Running_grads"][k]["Circular_window_buffer_voltage_a_sensitivity"][:,:,:,1:].clone()
            states['neurons']["Running_grads"][k]["Circular_window_buffer_voltage_a_sensitivity"][:,:,:,-1] = E_tot_a
            states['neurons']["Running_grads"][k]["Circular_window_buffer_voltage_b_sensitivity"][:,:,:,:-1] = states['neurons']["Running_grads"][k]["Circular_window_buffer_voltage_b_sensitivity"][:,:,:,1:].clone()
            states['neurons']["Running_grads"][k]["Circular_window_buffer_voltage_b_sensitivity"][:,:,:,-1] = E_tot_b
            states['neurons']["Running_grads"][k]["Circular_window_buffer_voltage_c_sensitivity"][:,:,:,:-1] = states['neurons']["Running_grads"][k]["Circular_window_buffer_voltage_c_sensitivity"][:,:,:,1:].clone()
            states['neurons']["Running_grads"][k]["Circular_window_buffer_voltage_c_sensitivity"][:,:,:,-1] = E_tot_c

            #Update adaptation accordingly. Both voltage and adpatation are udpated according to the gate and thier respective udpates within condition 2.
            adaptation_a[:,:,:,-1] = (adaptation_a[:,:,:,-2] + E_tot_a*states['neurons']['Learnable']['ron']["output_ad"][:,None,:])
            adaptation_a[:,:,:,-2] = adaptation_a[:,:,:,-1]
            adaptation_b[:,:,:,-1] = (adaptation_b[:,:,:,-2] + E_tot_b*states['neurons']['Learnable']['ron']["output_ad"][:,None,:])
            adaptation_b[:,:,:,-2] = adaptation_b[:,:,:,-1]
            adaptation_c[:,:,:,-1] = (adaptation_c[:,:,:,-2] + E_tot_c*states['neurons']['Learnable']['ron']["output_ad"][:,None,:])
            adaptation_c[:,:,:,-2] = adaptation_c[:,:,:,-1]

        else:
            mask = ((states['neurons']['Dynamic'][k]['V'][:,:,:,-1] >= states['neurons']['Static'][k]['V_thresh'])).to(torch.int64)
        if torch.any(mask).item():
            spikers = torch.where(mask)
            states['neurons']['Dynamic'][k]['tspike'][spikers + (states['neurons']['Dynamic'][k]['buffer_index'][spikers].to(torch.int64)-1,)] = timestep
            states['neurons']['Dynamic'][k]['buffer_index'][spikers] = states['neurons']['Dynamic'][k]['buffer_index'][spikers] % 5 + 1
            states['neurons']['Dynamic'][k]['V'][spikers + (-2,)] = states['neurons']['Dynamic'][k]['V'][spikers + (-1,)]
            states['neurons']['Dynamic'][k]['V'][spikers + (-1,)] = states['neurons']['Static'][k]['V_reset']
            states['neurons']['Dynamic'][k]['g_ad'][spikers + (-2,)] = states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)]
            states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)] = states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)]
            if states['neurons']['Static'][k]['output'] == 1:
                states['neurons']["Running_grads"][k]["output_ad_running"][spikers + (-2,)] = states['neurons']["Running_grads"][k]["output_ad_running"][spikers + (-1,)]
                states['neurons']["Running_grads"][k]["output_ad_running"][spikers + (-1,)] = states['neurons']["Running_grads"][k]["output_ad_running"][spikers + (-1,)] + 1
                states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)] = states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)] + states['neurons']['Learnable'][k]["output_ad"][spikers[0],None,spikers[2]].T
            else:
                states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)] = states['neurons']['Dynamic'][k]['g_ad'][spikers + (-1,)] + states['neurons']['Static'][k]['g_inc']

            #Sensitivities
            #Reset the sensitivies related to voltage -- preserve and use the previous timesteps sensitvities in the eligibility calcualtion later on
            for m in states['neurons']['Static'][k.split("_")[-1]]['projections']:
                cur_running_grads = states['synapses']['Running_grads'][m]
                cur_running_grads['total_running_contribution'][:,:,:,-1] = cur_running_grads['total_running_contribution'][:,:,:,-1]*(1-mask*1)

                for n in ['gain', 'alpha']:
                    cur_running_grads[f'total_running_contribution_{n}'][:,:,:,-1] = cur_running_grads[f'total_running_contribution_{n}'][:,:,:,-1]*(1-mask*1)

            if k == "on" or k == "off":
                cur_running_grads = states['neurons']['Running_grads'][k]
                for z in ['gain','alpha']:
                    cur_running_grads[f'total_running_contribution_{z}'][:,:,:,-1] = cur_running_grads[f'total_running_contribution_{z}'][:,:,:,-1]*(1-mask*1)

            if k == "ron":
                hidden_running_grads = states['synapses']['Running_grads']['sonoff_ron']
                for m in ["on", "off"]:
                    hidden_running_grads[f'hidden_running_contribution_{m}'][:,:,:,-1] = hidden_running_grads[f'hidden_running_contribution_{m}'][:,:,:,-1]*(1-mask*1)
                
                states['neurons']['Running_grads']['ron']["output_ad_contribution"][:,:,:,-1] = states['neurons']['Running_grads']['ron']["output_ad_contribution"][:,:,:,-1]*(1-mask*1)

        if k == 'ron' and timestep >= 5:
            delayed_mask = states['neurons']['Dynamic'][k]["spikes_holder"][..., timestep - 5].bool()
            if delayed_mask.any().item():
                spikers2 = torch.where((states['neurons']['Dynamic'][k]["spikes_holder"][..., timestep-5]))
                batch_idx, trial_idx, cell_idx = spikers2
                s_i = torch.sum(time_tensor*states['neurons']["Running_grads"][k]["Circular_window_buffer"][batch_idx,trial_idx,cell_idx,:],axis=-1)/torch.sum(states['neurons']["Running_grads"][k]["Circular_window_buffer"][batch_idx,trial_idx,cell_idx,:],axis=-1)
                s_i_rel = torch.sum(time_tensor*states['neurons']["Running_grads"][k]["Circular_window_buffer_rel"][batch_idx,trial_idx,cell_idx,:],axis=-1)/torch.sum(states['neurons']["Running_grads"][k]["Circular_window_buffer_rel"][batch_idx,trial_idx,cell_idx,:],axis=-1)
                window_sums = torch.sum((time_tensor-s_i[:,None])/torch.sum(states['neurons']["Running_grads"][k]["Circular_window_buffer"][batch_idx,trial_idx,cell_idx,:],axis=-1)[:,None]*states['neurons']["Running_grads"][k]["Circular_window_buffer_voltage_sensitivity"][batch_idx,trial_idx,cell_idx,:],axis=-1)
                window_sums_a = torch.sum((time_tensor-s_i_rel[:,None])/torch.sum(states['neurons']["Running_grads"][k]["Circular_window_buffer_rel"][batch_idx,trial_idx,cell_idx,:],axis=-1)[:,None]*states['neurons']["Running_grads"][k]["Circular_window_buffer_voltage_a_sensitivity"][batch_idx,trial_idx,cell_idx,:],axis=-1)
                window_sums_b = torch.sum((time_tensor-s_i_rel[:,None])/torch.sum(states['neurons']["Running_grads"][k]["Circular_window_buffer_rel"][batch_idx,trial_idx,cell_idx,:],axis=-1)[:,None]*states['neurons']["Running_grads"][k]["Circular_window_buffer_voltage_b_sensitivity"][batch_idx,trial_idx,cell_idx,:],axis=-1)
                window_sums_c = torch.sum((time_tensor-s_i_rel[:,None])/torch.sum(states['neurons']["Running_grads"][k]["Circular_window_buffer_rel"][batch_idx,trial_idx,cell_idx,:],axis=-1)[:,None]*states['neurons']["Running_grads"][k]["Circular_window_buffer_voltage_c_sensitivity"][batch_idx,trial_idx,cell_idx,:],axis=-1)
                states['neurons']["Running_grads"][k]["ds_dtheta"].append(window_sums)
                states['neurons']["Running_grads"][k]["ds_dtheta_a"].append(window_sums_a)
                states['neurons']["Running_grads"][k]["ds_dtheta_b"].append(window_sums_b)
                states['neurons']["Running_grads"][k]["ds_dtheta_c"].append(window_sums_c)
                states['neurons']["Running_grads"][k]["ds_dtheta_identity"].append([batch_idx,trial_idx,cell_idx])

                # if timestep == 28000:
                #     print('window_sums')
                #     print(window_sums_a)
                #     print('timestep')
                #     print(timestep)
    
    return states

def condtion3(args, states,timestep):
    for k in list(states['synapses']['Static']):
        pre_neuron_dynamic = states['neurons']['Dynamic'][k.split("_")[0]]
        mask = torch.any((timestep == ((pre_neuron_dynamic['tspike'] + int(states['synapses']['Static'][k]['PSC_delay']/args['simulation']['dt']))).to(torch.int64)),axis=-1)
        if torch.any(mask).item():
            spikers = torch.where(mask)
            states['synapses']['Dynamic'][k]['PSC_q'][spikers + (-2,)] = states['synapses']['Dynamic'][k]['PSC_q'][spikers + (-1,)]
            states['synapses']['Dynamic'][k]['PSC_x'][spikers + (-2,)] = states['synapses']['Dynamic'][k]['PSC_x'][spikers + (-1,)]
            states['synapses']['Dynamic'][k]['PSC_F'][spikers + (-2,)] = states['synapses']['Dynamic'][k]['PSC_F'][spikers + (-1,)]
            states['synapses']['Dynamic'][k]['PSC_P'][spikers + (-2,)] = states['synapses']['Dynamic'][k]['PSC_P'][spikers + (-1,)]
            # Match the July E-prop event ordering: x receives the previously
            # stored release q, then q is refreshed from the current F and P.
            states['synapses']['Dynamic'][k]['PSC_x'][spikers + (-1,)] = states['synapses']['Dynamic'][k]['PSC_x'][spikers + (-1,)] + states['synapses']['Dynamic'][k]['PSC_q'][spikers + (-1,)]
            states['synapses']['Dynamic'][k]['PSC_q'][spikers + (-1,)] = states['synapses']['Dynamic'][k]['PSC_F'][spikers + (-1,)] * states['synapses']['Dynamic'][k]['PSC_P'][spikers + (-1,)]
            states['synapses']['Dynamic'][k]['PSC_F'][spikers + (-1,)] = states['synapses']['Dynamic'][k]['PSC_F'][spikers + (-1,)] + states['synapses']['Static'][k]['PSC_fF']*(states['synapses']['Static'][k]['PSC_maxF']-states['synapses']['Dynamic'][k]['PSC_F'][spikers + (-1,)])
            states['synapses']['Dynamic'][k]['PSC_P'][spikers + (-1,)] = states['synapses']['Dynamic'][k]['PSC_P'][spikers + (-1,)] * (1-states['synapses']['Static'][k]['PSC_fP'])


            #Sensitivities
            #Gsyn
            if k == "sonoff_ron":
                for m in ["on","off"]:
                    cur_syn_static = states['synapses']['Static'][k]
                    cur_running_grads = states['synapses']['Running_grads'][k]

                    cur_running_grads[f'dPSCq{m}'][spikers + (-2,)] = cur_running_grads[f'dPSCq{m}'][spikers + (-1,)]
                    cur_running_grads[f'dPSCx{m}'][spikers + (-2,)] = cur_running_grads[f'dPSCx{m}'][spikers + (-1,)]
                    cur_running_grads[f'dPSCF{m}'][spikers + (-2,)] = cur_running_grads[f'dPSCF{m}'][spikers + (-1,)]
                    cur_running_grads[f'dPSCP{m}'][spikers + (-2,)] = cur_running_grads[f'dPSCP{m}'][spikers + (-1,)]
                    cur_running_grads[f'dPSCx{m}'][spikers + (-1,)] = cur_running_grads[f'dPSCx{m}'][spikers + (-1,)] + cur_running_grads[f'dPSCq{m}'][spikers + (-1,)]
                    cur_running_grads[f'dPSCq{m}'][spikers + (-1,)] = cur_running_grads[f'dPSCF{m}'][spikers + (-1,)] * states['synapses']['Dynamic'][k]['PSC_P'][spikers + (-2,)] + cur_running_grads[f'dPSCP{m}'][spikers + (-1,)] * states['synapses']['Dynamic'][k]['PSC_F'][spikers + (-2,)]
                    cur_running_grads[f'dPSCF{m}'][spikers + (-1,)] = cur_running_grads[f'dPSCF{m}'][spikers + (-1,)] - cur_syn_static['PSC_fF']*cur_running_grads[f'dPSCF{m}'][spikers + (-1,)]
                    cur_running_grads[f'dPSCP{m}'][spikers + (-1,)] = cur_running_grads[f'dPSCP{m}'][spikers + (-1,)] * (1-cur_syn_static['PSC_fP'])

            #STRF
            for m in ['gain','alpha']:
                cur_syn_static = states['synapses']['Static'][k]
                cur_running_grads = states['synapses']['Running_grads'][k]

                cur_running_grads[f'dPSCq{m}'][spikers + (-2,)] = cur_running_grads[f'dPSCq{m}'][spikers + (-1,)]
                cur_running_grads[f'dPSCx{m}'][spikers + (-2,)] = cur_running_grads[f'dPSCx{m}'][spikers + (-1,)]
                cur_running_grads[f'dPSCF{m}'][spikers + (-2,)] = cur_running_grads[f'dPSCF{m}'][spikers + (-1,)]
                cur_running_grads[f'dPSCP{m}'][spikers + (-2,)] = cur_running_grads[f'dPSCP{m}'][spikers + (-1,)]
                cur_running_grads[f'dPSCx{m}'][spikers + (-1,)] = cur_running_grads[f'dPSCx{m}'][spikers + (-1,)] + cur_running_grads[f'dPSCq{m}'][spikers + (-1,)]
                cur_running_grads[f'dPSCq{m}'][spikers + (-1,)] = cur_running_grads[f'dPSCF{m}'][spikers + (-1,)] * states['synapses']['Dynamic'][k]['PSC_P'][spikers + (-2,)] + cur_running_grads[f'dPSCP{m}'][spikers + (-1,)] * states['synapses']['Dynamic'][k]['PSC_F'][spikers + (-2,)]
                cur_running_grads[f'dPSCF{m}'][spikers + (-1,)] = cur_running_grads[f'dPSCF{m}'][spikers + (-1,)] - cur_syn_static['PSC_fF']*cur_running_grads[f'dPSCF{m}'][spikers + (-1,)]
                cur_running_grads[f'dPSCP{m}'][spikers + (-1,)] = cur_running_grads[f'dPSCP{m}'][spikers + (-1,)] * (1-cur_syn_static['PSC_fP'])


    return states
