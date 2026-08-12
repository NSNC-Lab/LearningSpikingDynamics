import torch

def run_odes(args, states, pre_processed_spike_object, pre_processed_rate_object,timestep):
    
    #Calculate projections for each neuron
    for k in list(states['neurons']['Static']):
        if timestep == 0:
                states['neurons']['Static'][k]['projections'] = caluclate_projections(states,k)

    #Run Adaptation eligbility
    k = "ron"
    cur_post_neuron_dynamic = states['neurons']['Dynamic'][k]
    cur_post_neuron_static = states['neurons']['Static'][k]  

    projection_portion = 0

    for m in states['neurons']['Static'][k]['projections']:
        projection_portion = projection_portion - cur_post_neuron_static['R']*states['synapses']['Dynamic'][m]['PSC_s'][:,:,:,-1]*states['synapses']['Learnable'][m]['gSYN'][:,None,:]
    
    voltage_transition = 1 + args['simulation']['dt']*(projection_portion - 
                                                                cur_post_neuron_static['R']*cur_post_neuron_static['nSYN']*cur_post_neuron_dynamic['noise_sn'][:,:,:,-1]
                                                                - 1 - cur_post_neuron_static['R']*cur_post_neuron_dynamic['g_ad'][:,:,:,-1])/cur_post_neuron_static['tau']

    states['neurons']["Running_grads"][k]["output_ad_contribution"][:,:,:,-1] = states['neurons']["Running_grads"][k]["output_ad_contribution"][:,:,:,-1] * voltage_transition + args['simulation']['dt']*-cur_post_neuron_static['R']*states['neurons']["Running_grads"][k]["output_ad_running"][:,:,:,-1]*(cur_post_neuron_dynamic['V'][:,:,:,-1]-cur_post_neuron_static['E_k'])/cur_post_neuron_static['tau']
    states['neurons']["Running_grads"][k]["output_ad_contribution"][:,:,:,-2] = states['neurons']["Running_grads"][k]["output_ad_contribution"][:,:,:,-1]

    
    for k in list(states['synapses']['Static']):

        #Run Gsyn Sensitivities
        cur_syn_dynamic = states['synapses']['Dynamic'][k]
        cur_syn_static = states['synapses']['Static'][k]
        
        cur_post_neuron_dynamic = states['neurons']['Dynamic'][k.split("_")[-1]]
        cur_post_neuron_static = states['neurons']['Static'][k.split("_")[-1]]

        
        cur_running_grads = states['synapses']['Running_grads'][k]
        cur_running_grads['total_running_contribution'][:,:,:,-2] = cur_running_grads['total_running_contribution'][:,:,:,-1]

        projection_portion = 0

        #sensitivites w.r.t gsyn
        for m in states['neurons']['Static'][k.split("_")[-1]]['projections']:
                projection_portion = projection_portion - cur_post_neuron_static['R']*states['synapses']['Dynamic'][m]['PSC_s'][:,:,:,-1]*states['synapses']['Learnable'][m]['gSYN'][:,None,:]

        if k.split("_")[1] == "ron":

            voltage_transition = 1 + args['simulation']['dt']*(projection_portion - 
                                                                cur_post_neuron_static['R']*cur_post_neuron_static['nSYN']*cur_post_neuron_dynamic['noise_sn'][:,:,:,-1]
                                                                - 1 - cur_post_neuron_static['R']*cur_post_neuron_dynamic['g_ad'][:,:,:,-1])/cur_post_neuron_static['tau']
            
            history_contribution = cur_running_grads['total_running_contribution'][:,:,:,-1] * voltage_transition


        if k.split("_")[1] == "sonoff":
            voltage_transition = 1 + args['simulation']['dt']*(projection_portion - 1 - cur_post_neuron_static['R']*cur_post_neuron_dynamic['g_ad'][:,:,:,-1])/cur_post_neuron_static['tau']
            history_contribution = cur_running_grads['total_running_contribution'][:,:,:,-1] * voltage_transition

        local_contribution =  args['simulation']['dt']*-cur_post_neuron_static['R']*cur_syn_dynamic['PSC_s'][:,:,:,-1]*(cur_post_neuron_dynamic['V'][:,:,:,-1]-cur_syn_static['ESYN'])/cur_post_neuron_static['tau']

        cur_running_grads['total_running_contribution'][:,:,:,-1] = history_contribution + local_contribution

        #Copy over now so you don't have to later
        cur_running_grads['total_running_contribution'][:,:,:,-2] = cur_running_grads['total_running_contribution'][:,:,:,-1]
 

        # Carry each hidden conductance through the sonoff -> ron PSC and into
        # a complete recursive output-voltage sensitivity.

        if k == "sonoff_ron":
            hidden_voltage_coefficient = (args['simulation']['dt'] * -cur_post_neuron_static['R'] * states['synapses']['Learnable'][k]['gSYN'][:,None,:] * (cur_post_neuron_dynamic['V'][:,:,:,-1] - cur_syn_static['ESYN']) / cur_post_neuron_static['tau'])

            for m in ["on", "off"]:
                hidden_voltage_trace = cur_running_grads[f'hidden_running_contribution_{m}']
                old_hidden_voltage = hidden_voltage_trace[:,:,:,-1].clone()
                hidden_voltage_trace[:,:,:,-1] = (voltage_transition * old_hidden_voltage + hidden_voltage_coefficient * cur_running_grads[f'dPSCs{m}'][:,:,:,-1])

                # Preserve the current pre-reset sensitivity for the
                # eligibility calculation that follows the conditionals.
                hidden_voltage_trace[:,:,:,-2] = hidden_voltage_trace[:,:,:,-1]



    #Run STRF sensitivities
    #The following covers all of the dv/dpsc branching along declared sysnapses
    for k in list(states['synapses']['Static']):

        cur_syn_dynamic = states['synapses']['Dynamic'][k]
        cur_syn_static = states['synapses']['Static'][k]        
        cur_post_neuron_dynamic = states['neurons']['Dynamic'][k.split("_")[-1]]
        cur_post_neuron_static = states['neurons']['Static'][k.split("_")[-1]]
        cur_running_grads = states['synapses']['Running_grads'][k]

        for z in ['gain', 'alpha']:  #First we need to seperate two seperate streams for our two different parameters

            if k.split("_")[1] == "ron":
                
                projection_portion = 0
                for m in states['neurons']['Static']['ron']['projections']:     #This is part of the jacobian update. dv_t+1/dv_t
                        projection_portion = projection_portion - cur_post_neuron_static['R']*states['synapses']['Dynamic'][m]['PSC_s'][:,:,:,-1]*states['synapses']['Learnable'][m]['gSYN'][:,None,:]

                ron_voltage_transition = 1 + args['simulation']['dt']*(projection_portion - 
                                                                        cur_post_neuron_static['R']*cur_post_neuron_static['nSYN']*cur_post_neuron_dynamic['noise_sn'][:,:,:,-1]
                                                                        - 1 - cur_post_neuron_static['R']*cur_post_neuron_dynamic['g_ad'][:,:,:,-1])/cur_post_neuron_static['tau']  #The rest of the jacobian update
                
                ron_history_contribution = cur_running_grads[f'total_running_contribution_{z}'][:,:,:,-1] * ron_voltage_transition #The chain that comes off of the jacobian, completing the following: dv_t+1/dv_t * dv_t/dpsc <- importantly dv_t/dpsc is just the previous value held by total_running_contribution.

                psc_local_ron = args['simulation']['dt']*-cur_post_neuron_static['R']*states['synapses']['Learnable'][k]['gSYN'][:,None,:]*cur_running_grads[f'dPSCs{z}'][:,:,:,-1]*(cur_post_neuron_dynamic['V'][:,:,:,-1]-cur_syn_static['ESYN'])/cur_post_neuron_static['tau'] #this is the direct part. So for the chain we will go dv/dpsc -> psc/spiking etc. so this is dv_t+1/dpsc_t directly.
                cur_running_grads[f'total_running_contribution_{z}'][:,:,:,-1] = ron_history_contribution + psc_local_ron # Add both contributions
                cur_running_grads[f'total_running_contribution_{z}'][:,:,:,-2] = cur_running_grads[f'total_running_contribution_{z}'][:,:,:,-1] #Update

            if k.split("_")[1] == "sonoff":
                 
                projection_portion = 0
                for m in states['neurons']['Static']['sonoff']['projections']:    
                    projection_portion = projection_portion - cur_post_neuron_static['R']*states['synapses']['Dynamic'][m]['PSC_s'][:,:,:,-1]*states['synapses']['Learnable'][m]['gSYN'][:,None,:]

                sonoff_voltage_transition = 1 + args['simulation']['dt']*(projection_portion - 1 - cur_post_neuron_static['R']*cur_post_neuron_dynamic['g_ad'][:,:,:,-1])/cur_post_neuron_static['tau']

                sonoff_history_contribution = cur_running_grads[f'total_running_contribution_{z}'][:,:,:,-1] * sonoff_voltage_transition

                psc_local_sonoff = args['simulation']['dt']*-cur_post_neuron_static['R']*states['synapses']['Learnable'][k]['gSYN'][:,None,:]*cur_running_grads[f'dPSCs{z}'][:,:,:,-1]*(cur_post_neuron_dynamic['V'][:,:,:,-1]-cur_syn_static['ESYN'])/cur_post_neuron_static['tau'] 
                cur_running_grads[f'total_running_contribution_{z}'][:,:,:,-1] = sonoff_history_contribution + psc_local_sonoff
                cur_running_grads[f'total_running_contribution_{z}'][:,:,:,-2] = cur_running_grads[f'total_running_contribution_{z}'][:,:,:,-1] 
    
    #This portion covers the dv/drate at the input. Samne idea as before, but now we are just targeting rate instead of psc.
    for k in ['on','off']:

        cur_neuron_dynamic = states['neurons']['Dynamic'][k]
        cur_neuron_static = states['neurons']['Static'][k]
        spks_string = f"{k}set_spks"  
        cur_running_grads = states['neurons']['Running_grads'][k]

        for z in ['gain', 'alpha']:
            
            voltage_transistion = 1 + args['simulation']['dt']*(-cur_neuron_static['R']*cur_neuron_static["g_postIC"]*pre_processed_spike_object['onset_offset_spks'][spks_string][:,:,:,timestep] - 1 - cur_neuron_static['R']*cur_neuron_dynamic['g_ad'][:,:,:,-1])/cur_neuron_static['tau']
            history_contribution = cur_running_grads[f'total_running_contribution_{z}'][:,:,:,-1] * voltage_transistion
            #Here I am seeding in the rate derivative for the STRFs. This connnects back the dV/dtheta, back to each parameter.
            if z == "gain":
                rate_str = "_gain"
            else:
                rate_str = ''

            #dt/1000 is from poisson conversion ratio
            local_rate_deriv =  (args['simulation']['dt']/1000)*args['simulation']['dt']*-cur_neuron_static['R']*cur_neuron_static["g_postIC"]*pre_processed_rate_object[f'{k}set_rate{rate_str}_deriv'][timestep,:,None,:]*(cur_neuron_dynamic['V'][:,:,:,-1]-cur_neuron_static['E_exc'])/cur_neuron_static['tau'] 

            cur_running_grads[f'total_running_contribution_{z}'][:,:,:,-1] = history_contribution + local_rate_deriv
            cur_running_grads[f'total_running_contribution_{z}'][:,:,:,-2] = cur_running_grads[f'total_running_contribution_{z}'][:,:,:,-1] 


    #Abs refractory sensitivity
    cur_post_neuron_static = states['neurons']['Static']['ron']
    cur_post_neuron_dynamic = states['neurons']['Dynamic']['ron']

    projection_portion = 0
    for m in states['neurons']['Static']['ron']['projections']:   
        projection_portion = projection_portion - cur_post_neuron_static['R']*states['synapses']['Dynamic'][m]['PSC_s'][:,:,:,-1]*states['synapses']['Learnable'][m]['gSYN'][:,None,:]
    
    ron_voltage_transition = 1 + args['simulation']['dt']*(projection_portion - 
                                                            cur_post_neuron_static['R']*cur_post_neuron_static['nSYN']*cur_post_neuron_dynamic['noise_sn'][:,:,:,-1]
                                                            - 1 - cur_post_neuron_static['R']*cur_post_neuron_dynamic['g_ad'][:,:,:,-1])/cur_post_neuron_static['tau'] 


    voltage_portion = states["neurons"]["Running_grads"]['ron']["abs_ref_voltage"]
    
    last_spike = states['neurons']['Dynamic']['ron']["tspike"].max(dim=-1).values
    a_t = (torch.tanh((timestep-last_spike)-states['neurons']['Learnable']['ron']['abs_ref'][:,None,:]/args['simulation']['dt']) + 1)/2
    a_t_prime = -(1 - torch.tanh((timestep-last_spike)-states['neurons']['Learnable']['ron']['abs_ref'][:,None,:]/args['simulation']['dt'])**2)/(2*args['simulation']['dt'])

    #This is the other portion of the jacobian  since the abs refractory period affects voltage
    voltage_portion[:,:,:,-1] = a_t*voltage_portion[:,:,:,-2]*ron_voltage_transition + a_t_prime*(states['neurons']['Dynamic']['ron']['V'][:,:,:,-1] - states['neurons']['Static']['ron']['V_reset'])
    voltage_portion[:,:,:,-2] = voltage_portion[:,:,:,-1]

    #Relative refracotory sensitivities (a,b,c)
    voltage_a = states["neurons"]["Running_grads"]['ron']["a_ref_voltage"]
    voltage_b = states["neurons"]["Running_grads"]['ron']["b_ref_voltage"]
    voltage_c = states["neurons"]["Running_grads"]['ron']["c_ref_voltage"]
    adaptation_a = states["neurons"]["Running_grads"]['ron']["a_ref_adaptation"]
    adaptation_b = states["neurons"]["Running_grads"]['ron']["b_ref_adaptation"]
    adaptation_c = states["neurons"]["Running_grads"]['ron']["c_ref_adaptation"]

    #Dt*Jt-Gt*Kt -> Rest of recursion is handled in condition handler
    voltage_a[:,:,:,-1] = ((voltage_a[:,:,:,-2]*ron_voltage_transition 
                           - adaptation_a[:,:,:,-2]*states['neurons']['Static']['ron']['R']*args['simulation']['dt']*(states['neurons']['Dynamic']['ron']['V'][:,:,:,-1] - states['neurons']['Static']['ron']['E_k'])/states['neurons']['Static']['ron']['tau']))
    voltage_b[:,:,:,-1] = ((voltage_b[:,:,:,-2]*ron_voltage_transition 
                               - adaptation_b[:,:,:,-2]*states['neurons']['Static']['ron']['R']*args['simulation']['dt']*(states['neurons']['Dynamic']['ron']['V'][:,:,:,-1] - states['neurons']['Static']['ron']['E_k'])/states['neurons']['Static']['ron']['tau']))
    voltage_c[:,:,:,-1] = ((voltage_c[:,:,:,-2]*ron_voltage_transition 
                                   - adaptation_c[:,:,:,-2]*states['neurons']['Static']['ron']['R']*args['simulation']['dt']*(states['neurons']['Dynamic']['ron']['V'][:,:,:,-1] - states['neurons']['Static']['ron']['E_k'])/states['neurons']['Static']['ron']['tau']))

    #Gt = Gt*Jat
    adaptation_jacobian = 1-args['simulation']['dt']/states['neurons']['Static']['ron']['tau_ad']
    adaptation_a[:,:,:,-1] = adaptation_jacobian*adaptation_a[:,:,:,-2]
    adaptation_b[:,:,:,-1] = adaptation_jacobian*adaptation_b[:,:,:,-2]
    adaptation_c[:,:,:,-1] = adaptation_jacobian*adaptation_c[:,:,:,-2]

    adaptation_a[:,:,:,-2] = adaptation_a[:,:,:,-1]
    adaptation_b[:,:,:,-2] = adaptation_b[:,:,:,-1]
    adaptation_c[:,:,:,-2] = adaptation_c[:,:,:,-1]
    voltage_a[:,:,:,-2] = voltage_a[:,:,:,-1]
    voltage_b[:,:,:,-2] = voltage_b[:,:,:,-1]
    voltage_c[:,:,:,-2] = voltage_c[:,:,:,-1]
                                   
    #Run Neuron ODES
    for k in list(states['neurons']['Static']):

        #Update Voltage
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

        if k == "ron":
            states['neurons']["Running_grads"][k]["output_ad_running"][:,:,:,-2] = states['neurons']["Running_grads"][k]["output_ad_running"][:,:,:,-1]
            states['neurons']["Running_grads"][k]["output_ad_running"][:,:,:,-1] = states['neurons']["Running_grads"][k]["output_ad_running"][:,:,:,-1] + (-states['neurons']["Running_grads"][k]["output_ad_running"][:,:,:,-1]/cur_static['tau_ad'])*args['simulation']['dt']


        cur_dynamic['g_ad'][:,:,:,-2] = cur_dynamic['g_ad'][:,:,:,-1]
        cur_dynamic['g_ad'][:,:,:,-1] = cur_dynamic['g_ad'][:,:,:,-1] + (-cur_dynamic['g_ad'][:,:,:,-1]/cur_static['tau_ad'])*args['simulation']['dt']

    #Run Synapse ODES
    for k in list(states['synapses']['Static']):

        cur_syn_dynamic = states['synapses']['Dynamic'][k]
        cur_syn_static = states['synapses']['Static'][k]
        
        cur_post_neuron_dynamic = states['neurons']['Dynamic'][k.split("_")[-1]]
        cur_post_neuron_static = states['neurons']['Static'][k.split("_")[-1]]

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

        #Backup pscs for eligbility handler
        cur_syn_dynamic['PSC_s'][:,:,:,-2] = cur_syn_dynamic['PSC_s'][:,:,:,-1]
        cur_syn_dynamic['PSC_x'][:,:,:,-2] = cur_syn_dynamic['PSC_x'][:,:,:,-1]
        cur_syn_dynamic['PSC_F'][:,:,:,-2] = cur_syn_dynamic['PSC_F'][:,:,:,-1]
        cur_syn_dynamic['PSC_P'][:,:,:,-2] = cur_syn_dynamic['PSC_P'][:,:,:,-1]
        cur_syn_dynamic['PSC_q'][:,:,:,-2] = cur_syn_dynamic['PSC_q'][:,:,:,-1]

        #These sensitivies will be updated in condition 3 which is why they can't just use the calculate values above... dPSC will evolve differently.
        if k == "sonoff_ron":
            for m in ["on","off"]:
                cur_syn_running = states['synapses']['Running_grads'][k]
                cur_syn_running[f'dPSCs{m}'][:,:,:,-2] = cur_syn_running[f'dPSCs{m}'][:,:,:,-1]
                cur_syn_running[f'dPSCx{m}'][:,:,:,-2] = cur_syn_running[f'dPSCx{m}'][:,:,:,-1]
                cur_syn_running[f'dPSCq{m}'][:,:,:,-2] = cur_syn_running[f'dPSCq{m}'][:,:,:,-1]
                cur_syn_running[f'dPSCF{m}'][:,:,:,-2] = cur_syn_running[f'dPSCF{m}'][:,:,:,-1]
                cur_syn_running[f'dPSCP{m}'][:,:,:,-2] = cur_syn_running[f'dPSCP{m}'][:,:,:,-1]

                cur_syn_running[f'dPSCs{m}'][:,:,:,-1] = cur_syn_running[f'dPSCs{m}'][:,:,:,-1] + args['simulation']['dt']*(cur_syn_static['scale']*cur_syn_running[f'dPSCx{m}'][:,:,:,-1] - cur_syn_running[f'dPSCs{m}'][:,:,:,-1])/cur_syn_static['tauR']
                cur_syn_running[f'dPSCx{m}'][:,:,:,-1] = cur_syn_running[f'dPSCx{m}'][:,:,:,-1] + args['simulation']['dt']*-cur_syn_running[f'dPSCx{m}'][:,:,:,-1]/cur_syn_static['tauD']
                cur_syn_running[f'dPSCq{m}'][:,:,:,-1] = cur_syn_running[f'dPSCq{m}'][:,:,:,-1] + args['simulation']['dt']*0
                cur_syn_running[f'dPSCF{m}'][:,:,:,-1] = cur_syn_running[f'dPSCF{m}'][:,:,:,-1] - args['simulation']['dt']*cur_syn_running[f'dPSCF{m}'][:,:,:,-1]/cur_syn_static['tauF']
                cur_syn_running[f'dPSCP{m}'][:,:,:,-1] = cur_syn_running[f'dPSCP{m}'][:,:,:,-1] - args['simulation']['dt']*cur_syn_running[f'dPSCP{m}'][:,:,:,-1]/cur_syn_static['tauP']
                
                cur_syn_running[f'dPSCs{m}'][:,:,:,-2] = cur_syn_running[f'dPSCs{m}'][:,:,:,-1]
                cur_syn_running[f'dPSCx{m}'][:,:,:,-2] = cur_syn_running[f'dPSCx{m}'][:,:,:,-1]
                cur_syn_running[f'dPSCq{m}'][:,:,:,-2] = cur_syn_running[f'dPSCq{m}'][:,:,:,-1]
                cur_syn_running[f'dPSCF{m}'][:,:,:,-2] = cur_syn_running[f'dPSCF{m}'][:,:,:,-1]
                cur_syn_running[f'dPSCP{m}'][:,:,:,-2] = cur_syn_running[f'dPSCP{m}'][:,:,:,-1]


        for m in ["gain","alpha"]:
            cur_syn_running = states['synapses']['Running_grads'][k]
            cur_syn_running[f'dPSCs{m}'][:,:,:,-2] = cur_syn_running[f'dPSCs{m}'][:,:,:,-1]
            cur_syn_running[f'dPSCx{m}'][:,:,:,-2] = cur_syn_running[f'dPSCx{m}'][:,:,:,-1]
            cur_syn_running[f'dPSCq{m}'][:,:,:,-2] = cur_syn_running[f'dPSCq{m}'][:,:,:,-1]
            cur_syn_running[f'dPSCF{m}'][:,:,:,-2] = cur_syn_running[f'dPSCF{m}'][:,:,:,-1]
            cur_syn_running[f'dPSCP{m}'][:,:,:,-2] = cur_syn_running[f'dPSCP{m}'][:,:,:,-1]

            cur_syn_running[f'dPSCs{m}'][:,:,:,-1] = cur_syn_running[f'dPSCs{m}'][:,:,:,-1] + args['simulation']['dt']*(cur_syn_static['scale']*cur_syn_running[f'dPSCx{m}'][:,:,:,-1] - cur_syn_running[f'dPSCs{m}'][:,:,:,-1])/cur_syn_static['tauR']
            cur_syn_running[f'dPSCx{m}'][:,:,:,-1] = cur_syn_running[f'dPSCx{m}'][:,:,:,-1] + args['simulation']['dt']*-cur_syn_running[f'dPSCx{m}'][:,:,:,-1]/cur_syn_static['tauD']
            cur_syn_running[f'dPSCq{m}'][:,:,:,-1] = cur_syn_running[f'dPSCq{m}'][:,:,:,-1] + args['simulation']['dt']*0
            cur_syn_running[f'dPSCF{m}'][:,:,:,-1] = cur_syn_running[f'dPSCF{m}'][:,:,:,-1] - args['simulation']['dt']*cur_syn_running[f'dPSCF{m}'][:,:,:,-1]/cur_syn_static['tauF']
            cur_syn_running[f'dPSCP{m}'][:,:,:,-1] = cur_syn_running[f'dPSCP{m}'][:,:,:,-1] - args['simulation']['dt']*cur_syn_running[f'dPSCP{m}'][:,:,:,-1]/cur_syn_static['tauP']
            
            cur_syn_running[f'dPSCs{m}'][:,:,:,-2] = cur_syn_running[f'dPSCs{m}'][:,:,:,-1]
            cur_syn_running[f'dPSCx{m}'][:,:,:,-2] = cur_syn_running[f'dPSCx{m}'][:,:,:,-1]
            cur_syn_running[f'dPSCq{m}'][:,:,:,-2] = cur_syn_running[f'dPSCq{m}'][:,:,:,-1]
            cur_syn_running[f'dPSCF{m}'][:,:,:,-2] = cur_syn_running[f'dPSCF{m}'][:,:,:,-1]
            cur_syn_running[f'dPSCP{m}'][:,:,:,-2] = cur_syn_running[f'dPSCP{m}'][:,:,:,-1]

        
    return states

def caluclate_projections(states,k):

    projections = []

    for m in list(states['synapses']['Static'].keys()):
        if m.split('_',-1)[-1] == k:
            projections.append(m)

    return projections
