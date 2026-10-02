import torch

def run_odes(args, states, pre_processed_spike_object,rate_object,timestep):

    #Put intial jacobian and trace estimates up here inside a nueron and synapse loop. Compile entire eligilbities once they can acutally be calcualted in conditional handler.

    for k in list(states['neurons']['Static']):

        if timestep == 0:
            states['neurons']['Static'][k]['projections'] = caluclate_projections(states,k)
        cur_dynamic = states['neurons']['Dynamic'][k]
        cur_static = states['neurons']['Static'][k]
        Jv_paths = 0

        if states['neurons']['Static'][k]['input'] == 1:
            spks_string = f"{k}set_spks"        
            Jv_paths = Jv_paths - cur_static['R']*cur_static['g_postIC']*pre_processed_spike_object['onset_offset_spks'][spks_string][:,:,:,timestep]
        if len(states['neurons']['Static'][k]['projections'])>0:
            for m in states['neurons']['Static'][k]['projections']:
                Jv_paths = Jv_paths - cur_static['R']*states['synapses']['Dynamic'][m]['PSC_s'][:,:,:,-1]*states['synapses']['Learnable'][m]['gSYN'][:,None,:]
            if states['neurons']['Static'][k]['noise'] == 1:
                Jv_paths = Jv_paths - cur_static['R']*cur_static['nSYN']*cur_dynamic['noise_sn'][:,:,:,-1]

        #JVV
        cur_dynamic['Jvv'][:,:,:,-2] = cur_dynamic['Jvv'][:,:,:,-1]
        cur_dynamic['Jvv'][:,:,:,-1] = (1 + (args['simulation']['dt']/cur_static['tau'])*(-1-cur_static['R']*cur_dynamic['g_ad'][:,:,:,-1]+Jv_paths))

        #JVA
        cur_dynamic['Jva'][:,:,:,-2] = cur_dynamic['Jva'][:,:,:,-1]
        cur_dynamic['Jva'][:,:,:,-1] = -cur_static['R']*(cur_dynamic['V'][:,:,:,-1]-cur_static['E_k'])*args['simulation']['dt']/cur_static['tau']

        #JAA
        cur_dynamic['Jaa'][:,:,:,-2] = cur_dynamic['Jaa'][:,:,:,-1]
        cur_dynamic['Jaa'][:,:,:,-1] = (1 - args['simulation']['dt']/cur_static['tau_ad'])
                                                

    for k in list(states['synapses']['Static']):
        cur_syn_dynamic = states['synapses']['Dynamic'][k]
        cur_syn_static = states['synapses']['Static'][k]
        cur_post_neuron_dynamic = states['neurons']['Dynamic'][k.split("_")[-1]]
        cur_post_neuron_static = states['neurons']['Static'][k.split("_")[-1]]

        #bt
        cur_syn_dynamic['bt'][:,:,:,-2] = cur_syn_dynamic['bt'][:,:,:,-1]
        cur_syn_dynamic['bt'][:,:,:,-1] = -(args['simulation']['dt']/cur_post_neuron_static['tau'])*cur_post_neuron_static['R']*cur_syn_dynamic['PSC_s'][:,:,:,-1]*(cur_post_neuron_dynamic['V'][:,:,:,-1]-cur_syn_static['ESYN'])


    #Run Neuron ODES
    
    for k in list(states['neurons']['Static']):

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
        
        #Udpate adaptation (scale factor for jacobian does not contain dynamic variables so it should be able to be included in the later udpate)
        cur_dynamic['g_ad'][:,:,:,-2] = cur_dynamic['g_ad'][:,:,:,-1]
        cur_dynamic['g_ad'][:,:,:,-1] = cur_dynamic['g_ad'][:,:,:,-1] + (-cur_dynamic['g_ad'][:,:,:,-1]/cur_static['tau_ad'])*args['simulation']['dt']

        #JAV and psi
        w=5
        cur_static['psi'] = (1 - torch.tanh((cur_dynamic['V'][:,:,:,-1] - cur_static['V_thresh'])/w)**2)/(2*w)
        cur_dynamic['Jav'][:,:,:,-2] = cur_dynamic['Jav'][:,:,:,-1]
        if states['neurons']['Static'][k]['output'] == 1:
            cur_dynamic['Jav'][:,:,:,-1] = cur_static['psi']*states['neurons']['Learnable'][k]["output_ad"][:,None,:]
        else:
            cur_dynamic['Jav'][:,:,:,-1] = cur_static['psi']*cur_static['g_inc']

    #Run Synapse ODES
    for k in list(states['synapses']['Static']):

        cur_syn_dynamic = states['synapses']['Dynamic'][k]
        cur_syn_static = states['synapses']['Static'][k]

        cur_post_neuron_dynamic = states['neurons']['Dynamic'][k.split("_")[-1]]
        cur_post_neuron_static = states['neurons']['Static'][k.split("_")[-1]]

        cur_syn_dynamic['epsilon_v'][:,:,:,-2] = cur_syn_dynamic['epsilon_v'][:,:,:,-1]
        cur_syn_dynamic['epsilon_v'][:,:,:,-1] = cur_post_neuron_dynamic['Jvv'][:,:,:,-1]*cur_syn_dynamic['epsilon_v'][:,:,:,-1] + cur_post_neuron_dynamic['Jva'][:,:,:,-1]*cur_syn_dynamic['epsilon_a'][:,:,:,-1] + cur_syn_dynamic['bt'][:,:,:,-1]

        cur_syn_dynamic['epsilon_a'][:,:,:,-2] = cur_syn_dynamic['epsilon_a'][:,:,:,-1]
        cur_syn_dynamic['epsilon_a'][:,:,:,-1] = cur_post_neuron_dynamic['Jaa'][:,:,:,-1]*cur_syn_dynamic['epsilon_a'][:,:,:,-1]

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

    #STRF Epsilon constructions
    on_static = states['neurons']["Static"]['on']
    on_dynamic = states['neurons']["Dynamic"]['on']
    off_static = states['neurons']["Static"]['off']
    off_dynamic = states['neurons']["Dynamic"]['off']

    states['neurons']["Dynamic"]['strf_gain_on']["epsilon_v"][:,:,:,-2] = states['neurons']["Dynamic"]['strf_gain_on']["epsilon_v"][:,:,:,-1]
    states['neurons']["Dynamic"]['strf_gain_off']["epsilon_v"][:,:,:,-2] = states['neurons']["Dynamic"]['strf_gain_off']["epsilon_v"][:,:,:,-1]
    states['neurons']["Dynamic"]['strf_gain_on']["epsilon_a"][:,:,:,-2] = states['neurons']["Dynamic"]['strf_gain_on']["epsilon_a"][:,:,:,-1]
    states['neurons']["Dynamic"]['strf_gain_off']["epsilon_a"][:,:,:,-2] = states['neurons']["Dynamic"]['strf_gain_off']["epsilon_a"][:,:,:,-1]
    states['neurons']["Dynamic"]['strf_alpha_on']["epsilon_v"][:,:,:,-2] = states['neurons']["Dynamic"]['strf_alpha_on']["epsilon_v"][:,:,:,-1]
    states['neurons']["Dynamic"]['strf_alpha_off']["epsilon_v"][:,:,:,-2] = states['neurons']["Dynamic"]['strf_alpha_off']["epsilon_v"][:,:,:,-1]
    states['neurons']["Dynamic"]['strf_alpha_on']["epsilon_a"][:,:,:,-2] = states['neurons']["Dynamic"]['strf_alpha_on']["epsilon_a"][:,:,:,-1]
    states['neurons']["Dynamic"]['strf_alpha_off']["epsilon_a"][:,:,:,-2] = states['neurons']["Dynamic"]['strf_alpha_off']["epsilon_a"][:,:,:,-1]

    states['neurons']["Dynamic"]['strf_gain_on']["epsilon_v"][:,:,:,-1] = on_dynamic['Jvv'][:,:,:,-1]*states['neurons']["Dynamic"]['strf_gain_on']['epsilon_v'][:,:,:,-1] + on_dynamic['Jva'][:,:,:,-1]*states['neurons']["Dynamic"]['strf_gain_on']['epsilon_a'][:,:,:,-1] - (args['simulation']['dt']/1000)*(args['simulation']['dt']/on_static['tau'])*on_static['R']*on_static['g_postIC']*rate_object['onset_rate_gain_deriv'][timestep,:,None,:]*(on_dynamic['V'][:,:,:,-2]-on_static['E_exc'])
    states['neurons']["Dynamic"]['strf_gain_off']["epsilon_v"][:,:,:,-1] = off_dynamic['Jvv'][:,:,:,-1]*states['neurons']["Dynamic"]['strf_gain_off']['epsilon_v'][:,:,:,-1] + off_dynamic['Jva'][:,:,:,-1]*states['neurons']["Dynamic"]['strf_gain_off']['epsilon_a'][:,:,:,-1] - (args['simulation']['dt']/1000)*(args['simulation']['dt']/off_static['tau'])*off_static['R']*off_static['g_postIC']*rate_object['offset_rate_gain_deriv'][timestep,:,None,:]*(off_dynamic['V'][:,:,:,-2]-off_static['E_exc'])
    states['neurons']["Dynamic"]['strf_gain_on']["epsilon_a"][:,:,:,-1] = on_dynamic['Jaa'][:,:,:,-1]*states['neurons']["Dynamic"]['strf_gain_on']['epsilon_a'][:,:,:,-1]
    states['neurons']["Dynamic"]['strf_gain_off']["epsilon_a"][:,:,:,-1] = off_dynamic['Jaa'][:,:,:,-1]*states['neurons']["Dynamic"]['strf_gain_off']['epsilon_a'][:,:,:,-1]
    states['neurons']["Dynamic"]['strf_alpha_on']["epsilon_v"][:,:,:,-1] = on_dynamic['Jvv'][:,:,:,-1]*states['neurons']["Dynamic"]['strf_alpha_on']['epsilon_v'][:,:,:,-1] + on_dynamic['Jva'][:,:,:,-1]*states['neurons']["Dynamic"]['strf_alpha_on']['epsilon_a'][:,:,:,-1] - (args['simulation']['dt']/1000)*(args['simulation']['dt']/on_static['tau'])*on_static['R']*on_static['g_postIC']*rate_object['onset_rate_deriv'][timestep,:,None,:]*(on_dynamic['V'][:,:,:,-2]-on_static['E_exc'])
    states['neurons']["Dynamic"]['strf_alpha_off']["epsilon_v"][:,:,:,-1] = off_dynamic['Jvv'][:,:,:,-1]*states['neurons']["Dynamic"]['strf_alpha_off']['epsilon_v'][:,:,:,-1] + off_dynamic['Jva'][:,:,:,-1]*states['neurons']["Dynamic"]['strf_alpha_off']['epsilon_a'][:,:,:,-1] - (args['simulation']['dt']/1000)*(args['simulation']['dt']/off_static['tau'])*off_static['R']*off_static['g_postIC']*rate_object['offset_rate_deriv'][timestep,:,None,:]*(off_dynamic['V'][:,:,:,-2]-off_static['E_exc'])
    states['neurons']["Dynamic"]['strf_alpha_on']["epsilon_a"][:,:,:,-1] = on_dynamic['Jaa'][:,:,:,-1]*states['neurons']["Dynamic"]['strf_alpha_on']['epsilon_a'][:,:,:,-1]
    states['neurons']["Dynamic"]['strf_alpha_off']["epsilon_a"][:,:,:,-1] = off_dynamic['Jaa'][:,:,:,-1]*states['neurons']["Dynamic"]['strf_alpha_off']['epsilon_a'][:,:,:,-1]

    return states

def caluclate_projections(states,k):

    projections = []

    for m in list(states['synapses']['Static'].keys()):
        if m.split('_',-1)[-1] == k:
            projections.append(m)

    return projections
