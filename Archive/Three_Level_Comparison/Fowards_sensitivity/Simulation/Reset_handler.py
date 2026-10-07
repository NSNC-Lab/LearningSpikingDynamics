import torch 

def reset_dyanmics(states,args,device):

    states["neurons"]["Learnable"]["STRF_gain_accum"] = torch.zeros((states["neurons"]["Learnable"]["STRF_gain"].shape), device=device, dtype=torch.float32)
    states["neurons"]["Learnable"]["STRF_gain_grad"] = torch.zeros((states["neurons"]["Learnable"]["STRF_gain"].shape), device=device, dtype=torch.float32) 
    states["neurons"]["Learnable"]["STRF_alpha_accum"] = torch.zeros((states["neurons"]["Learnable"]["STRF_alpha"].shape), device=device, dtype=torch.float32)
    states["neurons"]["Learnable"]["STRF_alpha_grad"] = torch.zeros((states["neurons"]["Learnable"]["STRF_alpha"].shape), device=device, dtype=torch.float32) 
    
    for name in list(states["neurons"]["Static"].keys()):
        states["neurons"]["Dynamic"][name]["V"] = torch.ones((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32) * states["neurons"]["Static"][name]["E_L"]
        states["neurons"]["Dynamic"][name]["g_ad"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        states["neurons"]["Dynamic"][name]["tspike"] = torch.ones((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),5), device=device, dtype=torch.float32) * -30
        states["neurons"]["Dynamic"][name]["buffer_index"] = torch.ones((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets'])), device=device, dtype=torch.int64)
        if states["neurons"]["Static"][name]["noise"] == 1:
            states["neurons"]["Dynamic"][name]["noise_sn"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
            states["neurons"]["Dynamic"][name]["noise_xn"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        if states["neurons"]["Static"][name]["output"] == 1:
            states["neurons"]["Dynamic"][name]["spikes_holder"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),args['simulation']['sim_len']), device=device, dtype=torch.int64)
            states["neurons"]["Dynamic"][name]["mean_sse_loss"] = 0
            states["neurons"]["Learnable"][name]["output_ad_accum"] = torch.zeros((states["neurons"]["Learnable"][name]["output_ad"].shape), device=device, dtype=torch.float32)
            states["neurons"]["Learnable"][name]["output_ad_grad"] = torch.zeros((states["neurons"]["Learnable"][name]["output_ad"].shape), device=device, dtype=torch.float32)
            states["neurons"]["Dynamic"][name]["mean_CV_loss"] = 0
            states["neurons"]["Learnable"][name]["abs_ref_accum"] = torch.zeros((states["neurons"]["Learnable"][name]["abs_ref"].shape), device=device, dtype=torch.float32)
            states["neurons"]["Learnable"][name]["abs_ref_grad"] = torch.zeros((states["neurons"]["Learnable"][name]["abs_ref"].shape), device=device, dtype=torch.float32)
            states["neurons"]["Learnable"][name]["rel_ref_a_accum"] = torch.zeros((states["neurons"]["Learnable"][name]["rel_ref_a"].shape), device=device, dtype=torch.float32)
            states["neurons"]["Learnable"][name]["rel_ref_a_grad"] = torch.zeros((states["neurons"]["Learnable"][name]["rel_ref_a"].shape), device=device, dtype=torch.float32)
            states["neurons"]["Learnable"][name]["rel_ref_b_accum"] = torch.zeros((states["neurons"]["Learnable"][name]["rel_ref_b"].shape), device=device, dtype=torch.float32)
            states["neurons"]["Learnable"][name]["rel_ref_b_grad"] = torch.zeros((states["neurons"]["Learnable"][name]["rel_ref_b"].shape), device=device, dtype=torch.float32)
            states["neurons"]["Learnable"][name]["rel_ref_c_accum"] = torch.zeros((states["neurons"]["Learnable"][name]["rel_ref_c"].shape), device=device, dtype=torch.float32)
            states["neurons"]["Learnable"][name]["rel_ref_c_grad"] = torch.zeros((states["neurons"]["Learnable"][name]["rel_ref_c"].shape), device=device, dtype=torch.float32)

            states["neurons"]["Running_grads"][name]["output_ad_running"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
            states["neurons"]["Running_grads"][name]["output_ad_contribution"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)

            states["neurons"]["Running_grads"][name]["abs_ref_voltage"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
            states["neurons"]["Running_grads"][name]["a_ref_voltage"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32) 
            states["neurons"]["Running_grads"][name]["b_ref_voltage"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32) 
            states["neurons"]["Running_grads"][name]["c_ref_voltage"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32) 

            states["neurons"]["Running_grads"][name]["a_ref_adaptation"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32) 
            states["neurons"]["Running_grads"][name]["b_ref_adaptation"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32) 
            states["neurons"]["Running_grads"][name]["c_ref_adaptation"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32) 

    for k in list(states["synapses"]["Static"].keys()):
        states["synapses"]["Dynamic"][k]["PSC_s"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        states["synapses"]["Dynamic"][k]["PSC_x"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        states["synapses"]["Dynamic"][k]["PSC_F"] = torch.ones((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        states["synapses"]["Dynamic"][k]["PSC_q"] = torch.ones((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        states["synapses"]["Dynamic"][k]["PSC_P"] = torch.ones((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        states["synapses"]["Learnable"][k]["gSYN_accum"] = torch.zeros((states["synapses"]["Learnable"][k]["gSYN"].shape), device=device, dtype=torch.float32)
        states["synapses"]["Learnable"][k]["gSYN_grad"] = torch.zeros((states["synapses"]["Learnable"][k]["gSYN"].shape), device=device, dtype=torch.float32)

        states["synapses"]["Running_grads"][k]["total_running_contribution"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)

        states["synapses"]["Running_grads"][k]["total_running_contribution"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)


        if k == "sonoff_ron":

            for m in ["on","off"]:
                states["synapses"]["Running_grads"][f'{m}_sonoff']["du_dg_circular"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),int(states["synapses"]["Running_grads"][f'{m}_sonoff']["du_dg_circular"].shape[-1])), device=device, dtype=torch.float32)

            for m in ["on","off"]:
                states["synapses"]["Running_grads"][k][f"hidden_running_contribution_{m}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
                states["synapses"]["Running_grads"][k][f"dPSCs{m}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
                states["synapses"]["Running_grads"][k][f"dPSCx{m}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
                states["synapses"]["Running_grads"][k][f"dPSCq{m}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
                states["synapses"]["Running_grads"][k][f"dPSCF{m}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
                states["synapses"]["Running_grads"][k][f"dPSCP{m}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)

        for m in ['gain','alpha']:
            states["synapses"]["Running_grads"][k][f"du_dg_circular_{m}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),int(states["synapses"]["Running_grads"][k][f"du_dg_circular_{m}"].shape[-1])), device=device, dtype=torch.float32)
            states["synapses"]["Running_grads"][k][f"total_running_contribution_{m}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
            states["synapses"]["Running_grads"][k][f"dPSCs{m}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
            states["synapses"]["Running_grads"][k][f"dPSCx{m}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
            states["synapses"]["Running_grads"][k][f"dPSCq{m}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
            states["synapses"]["Running_grads"][k][f"dPSCF{m}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
            states["synapses"]["Running_grads"][k][f"dPSCP{m}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
   
    for k in ['gain','alpha']:
        states["neurons"]["Running_grads"]["on"][f"total_running_contribution_{k}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        states["neurons"]["Running_grads"]['off'][f"total_running_contribution_{k}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)

    
    states["neurons"]["Running_grads"]['ron']["Circular_window_buffer"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),11), device=device, dtype=torch.float32)
    states["neurons"]["Running_grads"]['ron']["Circular_window_buffer_rel"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),11), device=device, dtype=torch.float32)
    states["neurons"]["Running_grads"]['ron']["Circular_window_buffer_voltage_sensitivity"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),11), device=device, dtype=torch.float32)
    states["neurons"]["Running_grads"]['ron']["Circular_window_buffer_voltage_a_sensitivity"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),11), device=device, dtype=torch.float32)
    states["neurons"]["Running_grads"]['ron']["Circular_window_buffer_voltage_b_sensitivity"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),11), device=device, dtype=torch.float32)
    states["neurons"]["Running_grads"]['ron']["Circular_window_buffer_voltage_c_sensitivity"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),11), device=device, dtype=torch.float32)
    states["neurons"]["Running_grads"]['ron']["ds_dtheta"] = []
    states["neurons"]["Running_grads"]['ron']["ds_dtheta_a"] = []
    states["neurons"]["Running_grads"]['ron']["ds_dtheta_b"] = []
    states["neurons"]["Running_grads"]['ron']["ds_dtheta_c"] = []
    states["neurons"]["Running_grads"]['ron']["ds_dtheta_identity"] = []

    return states
