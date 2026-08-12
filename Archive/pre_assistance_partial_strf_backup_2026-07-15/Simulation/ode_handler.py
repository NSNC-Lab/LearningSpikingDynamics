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

            if k == "ron":
                ron_dV_dV = (-1 - cur_static['R']*cur_dynamic['g_ad'][:,:,:,-1])/cur_static['tau']
                for m in states['neurons']['Static'][k]['projections']:
                    ron_dV_dV = ron_dV_dV - cur_static['R']*states['synapses']['Dynamic'][m]['PSC_s'][:,:,:,-1]*states['synapses']['Learnable'][m]['gSYN'][:,None,:]/cur_static['tau']
                if states['neurons']['Static'][k]['noise'] == 1:
                    ron_dV_dV = ron_dV_dV - cur_static['R']*cur_static['nSYN']*cur_dynamic['noise_sn'][:,:,:,-1]/cur_static['tau']
                for m in ("on_ron", "off_ron", "sonoff_ron"):
                    for param_name in ("gain", "alpha"):
                        syn_dynamic = states['synapses']['Dynamic'][m]
                        syn_static = states['synapses']['Static'][m]
                        syn_dynamic[f"dV_{param_name}"][:,:,:,-2] = syn_dynamic[f"dV_{param_name}"][:,:,:,-1]
                        syn_dynamic[f"dV_{param_name}"][:,:,:,-1] = syn_dynamic[f"dV_{param_name}"][:,:,:,-1] + args['simulation']['dt']*(ron_dV_dV*syn_dynamic[f"dV_{param_name}"][:,:,:,-1] - cur_static['R']*states['synapses']['Learnable'][m]['gSYN'][:,None,:]*(cur_dynamic['V'][:,:,:,-1]-syn_static['ESYN'])*syn_dynamic[f"dPSC_s_{param_name}"][:,:,:,-1]/cur_static['tau'])

            if k == "sonoff":
                sonoff_dV_dV = (-1 - cur_static['R']*cur_dynamic['g_ad'][:,:,:,-1])/cur_static['tau']
                for m in states['neurons']['Static'][k]['projections']:
                    sonoff_dV_dV = sonoff_dV_dV - cur_static['R']*states['synapses']['Dynamic'][m]['PSC_s'][:,:,:,-1]*states['synapses']['Learnable'][m]['gSYN'][:,None,:]/cur_static['tau']
                for m in ("on_sonoff", "off_sonoff"):
                    for param_name in ("gain", "alpha"):
                        syn_dynamic = states['synapses']['Dynamic'][m]
                        syn_static = states['synapses']['Static'][m]
                        syn_dynamic[f"dV_{param_name}"][:,:,:,-2] = syn_dynamic[f"dV_{param_name}"][:,:,:,-1]
                        syn_dynamic[f"dV_{param_name}"][:,:,:,-1] = syn_dynamic[f"dV_{param_name}"][:,:,:,-1] + args['simulation']['dt']*(sonoff_dV_dV*syn_dynamic[f"dV_{param_name}"][:,:,:,-1] - cur_static['R']*states['synapses']['Learnable'][m]['gSYN'][:,None,:]*(cur_dynamic['V'][:,:,:,-1]-syn_static['ESYN'])*syn_dynamic[f"dPSC_s_{param_name}"][:,:,:,-1]/cur_static['tau'])



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
        for param_name in ("gain", "alpha"):
            if f"dPSC_x_{param_name}" in cur_syn_dynamic:
                old_dPSC_s = cur_syn_dynamic[f"dPSC_s_{param_name}"][:,:,:,-1]
                old_dPSC_x = cur_syn_dynamic[f"dPSC_x_{param_name}"][:,:,:,-1]
                cur_syn_dynamic[f"dPSC_s_{param_name}"][:,:,:,-2] = old_dPSC_s
                cur_syn_dynamic[f"dPSC_x_{param_name}"][:,:,:,-2] = old_dPSC_x
                cur_syn_dynamic[f"dPSC_s_{param_name}"][:,:,:,-1] = old_dPSC_s + args['simulation']['dt']*(cur_syn_static['scale']*old_dPSC_x - old_dPSC_s)/cur_syn_static['tauR']
                cur_syn_dynamic[f"dPSC_x_{param_name}"][:,:,:,-1] = old_dPSC_x + args['simulation']['dt']*(-old_dPSC_x/cur_syn_static['tauD'])

    return states

def caluclate_projections(states,k):

    projections = []

    for m in list(states['synapses']['Static'].keys()):
        if m.split('_',-1)[-1] == k:
            projections.append(m)

    return projections
