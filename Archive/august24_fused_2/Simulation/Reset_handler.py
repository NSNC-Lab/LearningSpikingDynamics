import torch 

def reset_dyanmics(states,args,device):

    states["neurons"]["Learnable"]["STRF_gain_accum"] = torch.zeros((states["neurons"]["Learnable"]["STRF_gain"].shape), device=device, dtype=torch.float32)
    states["neurons"]["Learnable"]["STRF_gain_accum_rate"].zero_()
    states["neurons"]["Learnable"]["STRF_gain_grad"] = torch.zeros((states["neurons"]["Learnable"]["STRF_gain"].shape), device=device, dtype=torch.float32) 
    states["neurons"]["Learnable"]["STRF_alpha_accum"] = torch.zeros((states["neurons"]["Learnable"]["STRF_alpha"].shape), device=device, dtype=torch.float32)
    states["neurons"]["Learnable"]["STRF_alpha_accum_rate"].zero_()
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
            states["neurons"]["Learnable"][name]["output_ad_accum_rate"].zero_()
            states["neurons"]["Learnable"][name]["output_ad_trace"].zero_()
            states["neurons"]["Learnable"][name]["output_ad_grad"] = torch.zeros((states["neurons"]["Learnable"][name]["output_ad"].shape), device=device, dtype=torch.float32)
            states["neurons"]["Dynamic"][name]["mean_CV_loss"] = 0
            states["neurons"]["Learnable"][name]["abs_ref_accum"] = torch.zeros((states["neurons"]["Learnable"][name]["abs_ref"].shape), device=device, dtype=torch.float32)
            states["neurons"]["Learnable"][name]["abs_ref_accum_rate"].zero_()
            states["neurons"]["Learnable"][name]["abs_ref_grad"] = torch.zeros((states["neurons"]["Learnable"][name]["abs_ref"].shape), device=device, dtype=torch.float32)
            states["neurons"]["Learnable"][name]["rel_ref_a_accum"] = torch.zeros((states["neurons"]["Learnable"][name]["rel_ref_a"].shape), device=device, dtype=torch.float32)
            states["neurons"]["Learnable"][name]["rel_ref_a_accum_rate"].zero_()
            states["neurons"]["Learnable"][name]["rel_ref_a_grad"] = torch.zeros((states["neurons"]["Learnable"][name]["rel_ref_a"].shape), device=device, dtype=torch.float32)
            states["neurons"]["Learnable"][name]["rel_ref_b_accum"] = torch.zeros((states["neurons"]["Learnable"][name]["rel_ref_b"].shape), device=device, dtype=torch.float32)
            states["neurons"]["Learnable"][name]["rel_ref_b_accum_rate"].zero_()
            states["neurons"]["Learnable"][name]["rel_ref_b_grad"] = torch.zeros((states["neurons"]["Learnable"][name]["rel_ref_b"].shape), device=device, dtype=torch.float32)
            states["neurons"]["Learnable"][name]["rel_ref_c_accum"] = torch.zeros((states["neurons"]["Learnable"][name]["rel_ref_c"].shape), device=device, dtype=torch.float32)
            states["neurons"]["Learnable"][name]["rel_ref_c_accum_rate"].zero_()
            states["neurons"]["Learnable"][name]["rel_ref_c_grad"] = torch.zeros((states["neurons"]["Learnable"][name]["rel_ref_c"].shape), device=device, dtype=torch.float32)

    for k in list(states["synapses"]["Static"].keys()):
        states["synapses"]["Dynamic"][k]["PSC_s"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        states["synapses"]["Dynamic"][k]["PSC_x"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        states["synapses"]["Dynamic"][k]["PSC_F"] = torch.ones((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        states["synapses"]["Dynamic"][k]["PSC_P"] = torch.ones((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        states["synapses"]["Dynamic"][k]["PSC_q"] = torch.ones((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        states["synapses"]["Learnable"][k]["gSYN_accum"] = torch.zeros((states["synapses"]["Learnable"][k]["gSYN"].shape), device=device, dtype=torch.float32)
        states["synapses"]["Learnable"][k]["gSYN_accum_rate"].zero_()
        states["synapses"]["Learnable"][k]["gSYN_grad"] = torch.zeros((states["synapses"]["Learnable"][k]["gSYN"].shape), device=device, dtype=torch.float32)

    for name, value in states['neurons']['CV'].items():
        value.fill_(-1) if name == 'previous_time' else value.zero_()

    return states
