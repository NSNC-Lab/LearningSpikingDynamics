import torch

def run_odes(args, states, pre_processed_spike_object,timestep):
    
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

    return states

def caluclate_projections(states,k):

    projections = []

    for m in list(states['synapses']['Static'].keys()):
        if m.split('_',-1)[-1] == k:
            projections.append(m)

    return projections