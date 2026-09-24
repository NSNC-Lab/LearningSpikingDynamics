import torch
import numpy as np

def handle_loss(args, states, gt_data, timestep):

    if (timestep + 1) % args['simulation']['PSTH_granularity'] == 0 and timestep > 0:
        loss = calculate_loss(states, gt_data['psth_holder'], args, timestep)
        states = update_grad(states, loss['gradient'])

    if timestep == args['simulation']['sim_len'] - 1:
        loss_cv = calculate_CV_loss(states, gt_data['raster_holder'], args, timestep)
        states = update_grad_CV(states, loss_cv['gradient'])
        states = update_grad_rate(states, gt_data['psth_holder'], args, timestep,args["simulation"]["PSTH_granularity"])

    return states

def update_grad_rate(states, gt_data, args, timestep, granularity):
    # Calculate the gradient according to rate
    lamda = 10
    full_len = gt_data.shape[-1] * granularity
    exposure = 10 * full_len * args["simulation"]["dt"] / 1000

    sim_count = states["neurons"]["Dynamic"]["ron"]["spikes_holder"][..., :full_len].sum((1,3))
    target_count = gt_data.sum(-1)[None,:]

    sim_rate = sim_count / exposure
    target_rate = target_count / exposure
    gradient = 2 * lamda * (sim_rate-target_rate) / exposure

    #Update the parameters

    for k in list(states['synapses']['Static'].keys()): 
        if k.rsplit("_")[1] == "ron":
            states["synapses"]["Learnable"][k]["gSYN_grad"] += gradient*states["synapses"]["Learnable"][k]["gSYN_accum_rate"]
        else:
            states["synapses"]["Learnable"][k]["gSYN_grad"] += gradient*states["synapses"]["Learnable"][k]["gSYN_accum_rate"]
        states["synapses"]["Learnable"][k]["gSYN_accum_rate"].zero_()

    states["neurons"]["Learnable"]["ron"]["output_ad_grad"]+= gradient*states["neurons"]["Learnable"]["ron"]["output_ad_accum_rate"]
    states["neurons"]["Learnable"]["ron"]["output_ad_accum_rate"].zero_()

    states["neurons"]["Learnable"]["STRF_alpha_grad"] += gradient*states["neurons"]["Learnable"]["STRF_alpha_accum_rate"]
    states["neurons"]["Learnable"]["STRF_alpha_accum_rate"].zero_()

    states["neurons"]["Learnable"]["STRF_gain_grad"] += gradient*states["neurons"]["Learnable"]["STRF_gain_accum_rate"]
    states["neurons"]["Learnable"]["STRF_gain_accum_rate"].zero_()
    
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
    gradient = 2*(sim_psth - gt_data[None,:,target_index] - 0.5)
    return {'loss': loss, 'gradient': gradient}

def calculate_CV_loss(states,gt_data,args,timestep):

    #Calculate simualtion and data ISIs
    #Simulation
    gradient = torch.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets'])), device=torch.device(args['simulation']['device']), dtype=torch.float32)

    for k in range(args['simulation']['batch_size']):
        for m in range(len(args['simulation']['cell_targets'])):
            isi_holder_sim = []
            isi_holder_data = []
            for n in range(10):
                if len(torch.where(states["neurons"]["Dynamic"]['ron']['spikes_holder'][k,n,m,:] == 1)[0]) > 0:
                    isi_holder_sim.append(torch.diff(torch.where(states["neurons"]["Dynamic"]['ron']['spikes_holder'][k,n,m,:] == 1)[0]))
                
                isi_holder_data.append(torch.diff(torch.where(gt_data[m,n,:] == 1)[0]))

            
            if len(isi_holder_sim) <= 1: #The idea behind this is that if there is nothing to go off of (aka no spikes) don't move the gradient... The other loss should rebound things (assuming that the refractory parameters are well enough constrained)
                gradient[k,m] = 0
            elif torch.mean(torch.cat(isi_holder_sim).to(torch.float32)) == 0 or len(torch.cat(isi_holder_sim).to(torch.float32)) <= 1: #If mean is zero it will break the grad, or if there is only 1 example it will break the std.
                gradient[k,m] = 0
            else:
                sim_cv = torch.std(torch.cat(isi_holder_sim).to(torch.float32))/torch.mean(torch.cat(isi_holder_sim).to(torch.float32))
                data_cv = torch.std(torch.cat(isi_holder_data).to(torch.float32))/torch.mean(torch.cat(isi_holder_data).to(torch.float32))

                loss = (sim_cv - data_cv)**2
                states["neurons"]["Dynamic"]['ron']['mean_CV_loss'] += loss

                lamda2 = 2
                gradient[k,m] = 2*(sim_cv - data_cv)*lamda2

    return {'loss': loss, 'gradient': gradient}    

def update_grad(states,gradient):
    for k in list(states['synapses']['Static'].keys()): 
        if k.rsplit("_")[1] == "ron":
            states["synapses"]["Learnable"][k]["gSYN_grad"] += gradient*states["synapses"]["Learnable"][k]["gSYN_accum"]
            states["synapses"]["Learnable"][k]["gSYN_accum_rate"] += states["synapses"]["Learnable"][k]["gSYN_accum"]
        else:
            states["synapses"]["Learnable"][k]["gSYN_grad"] += gradient*states["synapses"]["Learnable"][k]["gSYN_accum"]*states['neurons']['Learnable']['Bk']
            states["synapses"]["Learnable"][k]["gSYN_accum_rate"] += states["synapses"]["Learnable"][k]["gSYN_accum"]*states['neurons']['Learnable']['Bk']
        states["synapses"]["Learnable"][k]["gSYN_accum"].zero_()

    states["neurons"]["Learnable"]["ron"]["output_ad_grad"]+= gradient*states["neurons"]["Learnable"]["ron"]["output_ad_accum"]
    states["neurons"]["Learnable"]["ron"]["output_ad_accum_rate"] += states["neurons"]["Learnable"]["ron"]["output_ad_accum"]
    states["neurons"]["Learnable"]["ron"]["output_ad_accum"].zero_()

    states["neurons"]["Learnable"]["STRF_alpha_grad"] += gradient*states["neurons"]["Learnable"]["STRF_alpha_accum"]
    states["neurons"]["Learnable"]["STRF_alpha_accum_rate"] += states["neurons"]["Learnable"]["STRF_alpha_accum"]
    states["neurons"]["Learnable"]["STRF_alpha_accum"].zero_()

    states["neurons"]["Learnable"]["STRF_gain_grad"] += gradient*states["neurons"]["Learnable"]["STRF_gain_accum"]
    states["neurons"]["Learnable"]["STRF_gain_accum_rate"] += states["neurons"]["Learnable"]["STRF_gain_accum"]
    states["neurons"]["Learnable"]["STRF_gain_accum"].zero_()

    return states

def update_grad_CV(states,gradient):

    states["neurons"]["Learnable"]["ron"]["abs_ref_grad"] += gradient*states["neurons"]["Learnable"]["ron"]["abs_ref_accum"]
    states["neurons"]["Learnable"]["ron"]["rel_ref_a_grad"] += gradient*states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum"]
    states["neurons"]["Learnable"]["ron"]["rel_ref_b_grad"] += gradient*states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum"]
    states["neurons"]["Learnable"]["ron"]["rel_ref_c_grad"] += gradient*states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum"]

    return states
