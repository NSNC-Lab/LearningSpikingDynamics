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
        states["neurons"]["Dynamic"][name]["spike_mask"] = torch.zeros_like(
            states["neurons"]["Dynamic"][name]["spike_mask"]
        )
        if states["neurons"]["Static"][name]["input"] == 1:
            for param_name in ("gain", "alpha"):
                trace_name = f"dV_STRF_{param_name}"
                states["neurons"]["Dynamic"][name][trace_name] = torch.zeros_like(states["neurons"]["Dynamic"][name][trace_name])
            states["neurons"]["Dynamic"][name]["input_V_jacobian"] = torch.full_like(
                states["neurons"]["Dynamic"][name]["input_V_jacobian"],
                -1.0 / states["neurons"]["Static"][name]["tau"],
            )
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
            states["neurons"]["Dynamic"][name]["dV_output_ad"] = torch.zeros_like(states["neurons"]["Dynamic"][name]["dV_output_ad"])
            states["neurons"]["Dynamic"][name]["dg_ad_output_ad"] = torch.zeros_like(states["neurons"]["Dynamic"][name]["dg_ad_output_ad"])

    for k in list(states["synapses"]["Static"].keys()):
        states["synapses"]["Dynamic"][k]["PSC_s"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        states["synapses"]["Dynamic"][k]["PSC_x"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        states["synapses"]["Dynamic"][k]["PSC_F"] = torch.ones((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        states["synapses"]["Dynamic"][k]["PSC_q"] = torch.ones((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        states["synapses"]["Dynamic"][k]["PSC_P"] = torch.ones((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
        states["synapses"]["Dynamic"][k]["spike_delay_queue"] = torch.zeros_like(
            states["synapses"]["Dynamic"][k]["spike_delay_queue"]
        )
        if "dV_gSYN" in states["synapses"]["Dynamic"][k]:
            states["synapses"]["Dynamic"][k]["dV_gSYN"] = torch.zeros_like(states["synapses"]["Dynamic"][k]["dV_gSYN"])
        for param_name in ("gain", "alpha"):
            if f"dPSC_x_{param_name}" in states["synapses"]["Dynamic"][k]:
                states["synapses"]["Dynamic"][k][f"dPSC_F_{param_name}"] = torch.zeros_like(states["synapses"]["Dynamic"][k][f"dPSC_F_{param_name}"])
                states["synapses"]["Dynamic"][k][f"dPSC_P_{param_name}"] = torch.zeros_like(states["synapses"]["Dynamic"][k][f"dPSC_P_{param_name}"])
                states["synapses"]["Dynamic"][k][f"dPSC_x_{param_name}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
                states["synapses"]["Dynamic"][k][f"dPSC_s_{param_name}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
                states["synapses"]["Dynamic"][k][f"dV_{param_name}"] = torch.zeros((args['simulation']['batch_size'],10,len(args['simulation']['cell_targets']),2), device=device, dtype=torch.float32)
                delay_name = (
                    f"dtime_{param_name}_delay"
                    if k == "sonoff_ron"
                    else f"dsoft_{param_name}_delay"
                )
                states["synapses"]["Dynamic"][k][delay_name] = torch.zeros_like(states["synapses"]["Dynamic"][k][delay_name])
        if k == "sonoff_ron":
            for source_name in ("on_sonoff", "off_sonoff"):
                trace_name = f"{source_name}_gSYN"
                for state_name in (
                    f"dPSC_F_{trace_name}",
                    f"dPSC_P_{trace_name}",
                    f"dPSC_x_{trace_name}",
                    f"dPSC_s_{trace_name}",
                    f"dV_{trace_name}",
                    f"dtime_{trace_name}_delay",
                ):
                    states["synapses"]["Dynamic"][k][state_name] = torch.zeros_like(states["synapses"]["Dynamic"][k][state_name])
        states["synapses"]["Learnable"][k]["gSYN_accum"] = torch.zeros((states["synapses"]["Learnable"][k]["gSYN"].shape), device=device, dtype=torch.float32)
        states["synapses"]["Learnable"][k]["gSYN_grad"] = torch.zeros((states["synapses"]["Learnable"][k]["gSYN"].shape), device=device, dtype=torch.float32)

    return states
