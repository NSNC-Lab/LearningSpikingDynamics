import torch 

def run_adam(states, args, lrs):

    eps = 1e-8

    #Update t
    states["neurons"]["Adam"]["t"] += 1

    #Update m
    states["neurons"]["Adam"]["m"][:,:,0] = args['Adam']['beta1']*states["neurons"]["Adam"]["m"][:,:,0] + (1-args['Adam']['beta1'])*states["neurons"]["Learnable"]["STRF_gain_grad"]
    states["neurons"]["Adam"]["m"][:,:,1] = args['Adam']['beta1']*states["neurons"]["Adam"]["m"][:,:,1] + (1-args['Adam']['beta1'])*states["neurons"]["Learnable"]["STRF_alpha_grad"]
    states["neurons"]["Adam"]["m"][:,:,2] = args['Adam']['beta1']*states["neurons"]["Adam"]["m"][:,:,2] + (1-args['Adam']['beta1'])*states["neurons"]["Learnable"]["ron"]["output_ad_grad"]
    for val,k in enumerate(list(states["synapses"]["Static"].keys())):
        states["neurons"]["Adam"]["m"][:,:,val+3] = args['Adam']['beta1']*states["neurons"]["Adam"]["m"][:,:,val+3] + (1-args['Adam']['beta1'])*states["synapses"]["Learnable"][k]["gSYN_grad"]
    
    states["neurons"]["Adam"]["m"][:,:,8] = args['Adam']['beta1']*states["neurons"]["Adam"]["m"][:,:,8] + (1-args['Adam']['beta1'])*states["neurons"]["Learnable"]["ron"]["abs_ref_grad"]
    states["neurons"]["Adam"]["m"][:,:,9] = args['Adam']['beta1']*states["neurons"]["Adam"]["m"][:,:,9] + (1-args['Adam']['beta1'])*states["neurons"]["Learnable"]["ron"]["rel_ref_a_grad"]
    states["neurons"]["Adam"]["m"][:,:,10] = args['Adam']['beta1']*states["neurons"]["Adam"]["m"][:,:,10] + (1-args['Adam']['beta1'])*states["neurons"]["Learnable"]["ron"]["rel_ref_b_grad"]
    states["neurons"]["Adam"]["m"][:,:,11] = args['Adam']['beta1']*states["neurons"]["Adam"]["m"][:,:,11] + (1-args['Adam']['beta1'])*states["neurons"]["Learnable"]["ron"]["rel_ref_c_grad"]

    #Update v
    states["neurons"]["Adam"]["v"][:,:,0] = args['Adam']['beta2']*states["neurons"]["Adam"]["v"][:,:,0] + (1-args['Adam']['beta2'])*(states["neurons"]["Learnable"]["STRF_gain_grad"]**2)
    states["neurons"]["Adam"]["v"][:,:,1] = args['Adam']['beta2']*states["neurons"]["Adam"]["v"][:,:,1] + (1-args['Adam']['beta2'])*(states["neurons"]["Learnable"]["STRF_alpha_grad"]**2)
    states["neurons"]["Adam"]["v"][:,:,2] = args['Adam']['beta2']*states["neurons"]["Adam"]["v"][:,:,2] + (1-args['Adam']['beta2'])*(states["neurons"]["Learnable"]["ron"]["output_ad_grad"]**2)
    for val,k in enumerate(list(states["synapses"]["Static"].keys())):
        states["neurons"]["Adam"]["v"][:,:,val+3] = args['Adam']['beta2']*states["neurons"]["Adam"]["v"][:,:,val+3] + (1-args['Adam']['beta2'])*(states["synapses"]["Learnable"][k]["gSYN_grad"]**2)

    states["neurons"]["Adam"]["v"][:,:,8] = args['Adam']['beta2']*states["neurons"]["Adam"]["v"][:,:,8] + (1-args['Adam']['beta2'])*(states["neurons"]["Learnable"]["ron"]["abs_ref_grad"]**2)
    states["neurons"]["Adam"]["v"][:,:,9] = args['Adam']['beta2']*states["neurons"]["Adam"]["v"][:,:,9] + (1-args['Adam']['beta2'])*(states["neurons"]["Learnable"]["ron"]["rel_ref_a_grad"]**2)
    states["neurons"]["Adam"]["v"][:,:,10] = args['Adam']['beta2']*states["neurons"]["Adam"]["v"][:,:,10] + (1-args['Adam']['beta2'])*(states["neurons"]["Learnable"]["ron"]["rel_ref_b_grad"]**2)
    states["neurons"]["Adam"]["v"][:,:,11] = args['Adam']['beta2']*states["neurons"]["Adam"]["v"][:,:,11] + (1-args['Adam']['beta2'])*(states["neurons"]["Learnable"]["ron"]["rel_ref_c_grad"]**2)

    m_hat = states["neurons"]["Adam"]["m"]/(1-args['Adam']['beta1']**states["neurons"]["Adam"]["t"])
    v_hat = states["neurons"]["Adam"]["v"]/(1-args['Adam']['beta2']**states["neurons"]["Adam"]["t"])

    #Update the parameters
    states["neurons"]["Learnable"]["STRF_gain"] = states["neurons"]["Learnable"]["STRF_gain"] - (lrs['Strf_gain']*m_hat[:,:,0])/(torch.sqrt(v_hat[:,:,0]) + eps)
    states["neurons"]["Learnable"]["STRF_alpha"] = states["neurons"]["Learnable"]["STRF_alpha"] - (lrs['Strf_alpha']*m_hat[:,:,1])/(torch.sqrt(v_hat[:,:,1]) + eps)
    states["neurons"]["Learnable"]["ron"]["output_ad"] = states["neurons"]["Learnable"]["ron"]["output_ad"] - (lrs['output_ad']*m_hat[:,:,2])/(torch.sqrt(v_hat[:,:,2]) + eps)
    for val,k in enumerate(list(states["synapses"]["Static"].keys())):
        states["synapses"]["Learnable"][k]["gSYN"] = states["synapses"]["Learnable"][k]["gSYN"] - (lrs[f'{k}_gSYN']*m_hat[:,:,val+3])/(torch.sqrt(v_hat[:,:,val+3]) + eps)

    states["neurons"]["Learnable"]["ron"]["abs_ref"] = states["neurons"]["Learnable"]["ron"]["abs_ref"] - (lrs['abs_ref']*m_hat[:,:,8])/(torch.sqrt(v_hat[:,:,8]) + eps)
    states["neurons"]["Learnable"]["ron"]["rel_ref_a"] = states["neurons"]["Learnable"]["ron"]["rel_ref_a"] - (lrs['rel_ref_a']*m_hat[:,:,9])/(torch.sqrt(v_hat[:,:,9]) + eps)
    states["neurons"]["Learnable"]["ron"]["rel_ref_b"] = states["neurons"]["Learnable"]["ron"]["rel_ref_b"] - (lrs['rel_ref_b']*m_hat[:,:,10])/(torch.sqrt(v_hat[:,:,10]) + eps)
    states["neurons"]["Learnable"]["ron"]["rel_ref_c"] = states["neurons"]["Learnable"]["ron"]["rel_ref_c"] - (lrs['rel_ref_c']*m_hat[:,:,11])/(torch.sqrt(v_hat[:,:,11]) + eps)

    states["neurons"]["Learnable"]["STRF_gain"] = torch.clamp(states["neurons"]["Learnable"]["STRF_gain"],min=0)
    states["neurons"]["Learnable"]["STRF_alpha"] = torch.clamp(states["neurons"]["Learnable"]["STRF_alpha"],min=5,max=250)
    states["neurons"]["Learnable"]["ron"]["output_ad"] = torch.clamp(states["neurons"]["Learnable"]["ron"]["output_ad"],min=0)
    for val,k in enumerate(list(states["synapses"]["Static"].keys())):
        states["synapses"]["Learnable"][k]["gSYN"] = torch.clamp(states["synapses"]["Learnable"][k]["gSYN"],min=0)
    states["neurons"]["Learnable"]["ron"]["abs_ref"] = torch.clamp(states["neurons"]["Learnable"]["ron"]["abs_ref"],min=0)
    states["neurons"]["Learnable"]["ron"]["rel_ref_a"] = torch.clamp(states["neurons"]["Learnable"]["ron"]["rel_ref_a"],min=0)
    states["neurons"]["Learnable"]["ron"]["rel_ref_b"] = torch.clamp(states["neurons"]["Learnable"]["ron"]["rel_ref_b"],min=0)
    states["neurons"]["Learnable"]["ron"]["rel_ref_c"] = torch.clamp(states["neurons"]["Learnable"]["ron"]["rel_ref_c"],min=0.2,max=1)

    return states