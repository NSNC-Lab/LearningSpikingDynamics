import torch
import numpy as np

def handle_loss(args, states, gt_data, timestep):

    if (timestep + 1) % args['simulation']['PSTH_granularity'] == 0 and timestep > 0:
        loss = calculate_loss(states, gt_data['psth_holder'], args, timestep)
        states = update_grad(states, loss['gradient'])
    return states

def calculate_loss(states,gt_data,args,timestep):

    #Caluclate simulation PSTHs
    #Trim 
    bin_end = timestep + 1
    bin_start = bin_end - args["simulation"]["PSTH_granularity"]
    target_index = bin_end // args["simulation"]["PSTH_granularity"] - 1
    holder = states["neurons"]["Dynamic"]['ron']['spikes_holder'][:,:,:,bin_start:bin_end]
    #Sums across trial dim
    holder = torch.sum(holder, dim = 1,keepdim = False)
    #Reshape in order to sum across bins
    #holder = holder.reshape((args['simulation']['batch_size'],len(args['simulation']['cell_targets']),int(np.round(args["simulation"]["sim_len"]/args["simulation"]["PSTH_granularity"])),args["simulation"]["PSTH_granularity"]))
    sim_psth = torch.sum(holder, dim = -1,keepdim = False)

    #Calculate Loss and gradient
    #Average SSE
    loss = (sim_psth - gt_data[None,:,target_index])**2
    states["neurons"]["Dynamic"]['ron']['mean_sse_loss'] += torch.mean(loss.flatten()).cpu()

    #Associated SSE gradient
    gradient = 2*(sim_psth - gt_data[None,:,target_index])
    return {'loss': loss, 'gradient': gradient}


def update_grad(states,gradient):
    for k in list(states['synapses']['Static'].keys()): 
        if k.rsplit("_")[1] == "ron":
            states["synapses"]["Learnable"][k]["gSYN_grad"] += gradient*states["synapses"]["Learnable"][k]["gSYN_accum"]
        else:
            states["synapses"]["Learnable"][k]["gSYN_grad"] += gradient*states["synapses"]["Learnable"][k]["gSYN_accum"]*states['neurons']['Learnable']['Bk']
        states["synapses"]["Learnable"][k]["gSYN_accum"].zero_()

    states["neurons"]["Learnable"]["ron"]["output_ad_grad"]+= gradient*states["neurons"]["Learnable"]["ron"]["output_ad_accum"]
    states["neurons"]["Learnable"]["ron"]["output_ad_accum"].zero_()

    states["neurons"]["Learnable"]["STRF_alpha_grad"] += gradient*states["neurons"]["Learnable"]["STRF_alpha_accum"]
    states["neurons"]["Learnable"]["STRF_alpha_accum"].zero_()

    states["neurons"]["Learnable"]["STRF_gain_grad"] += gradient*states["neurons"]["Learnable"]["STRF_gain_accum"]
    states["neurons"]["Learnable"]["STRF_gain_accum"].zero_()

    return states

