import torch
import numpy as np

def handle_loss(args, states, gt_data, timestep):

    if (timestep + 1) % args['simulation']['PSTH_granularity'] == 0 and timestep > 0:
        loss = calculate_loss(states, gt_data['psth_holder'], args, timestep)
        lam = 5.0
        width = args['simulation']['PSTH_granularity']
        bins = args['simulation']['sim_len'] // width
        output = states['neurons']['Dynamic']['ron']
        if timestep + 1 == width:
            output['rate_error'] = torch.zeros_like(loss['gradient'])
        output['rate_error'] += loss['gradient'] / 2
        # Preserve feedback-weighted eligibility before update_grad clears each bin.
        learnable = states['neurons']['Learnable']
        groups = [(states['synapses']['Learnable'][k], 1 if k.rsplit('_', 1)[-1] == 'ron' else learnable['Bk'])
                  for k in states['synapses']['Static']] + [(learnable, 1), (learnable['ron'], 1)]
        for group, feedback in groups:
            for key in [k for k in group if k.endswith('_accum')]:
                if timestep + 1 == width:
                    group[key + '_rate'] = torch.zeros_like(group[key])
                group[key + '_rate'] += group[key] * feedback
                if timestep + 1 == bins * width:
                    # L_rate = lam / bins * (sum of bin-count errors)**2.
                    group[key[:-6] + '_grad'] += 2 * lam * output['rate_error'] / bins * group[key + '_rate']
        states = update_grad(states, loss['gradient'])

    if timestep == args['simulation']['sim_len'] - 1:
        loss_cv = calculate_CV_loss(states, gt_data['raster_holder'], args, timestep)
        states = update_grad_CV(states, loss_cv['gradient'])

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

    states["neurons"]["Learnable"]["ron"]["abs_ref_grad"] += gradient*states["neurons"]["Learnable"]["ron"]["abs_ref_accum"]
    states["neurons"]["Learnable"]["ron"]["abs_ref_accum"].zero_()
    states["neurons"]["Learnable"]["ron"]["rel_ref_a_grad"] += gradient*states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum"]
    states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum"].zero_()
    states["neurons"]["Learnable"]["ron"]["rel_ref_b_grad"] += gradient*states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum"]
    states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum"].zero_()
    states["neurons"]["Learnable"]["ron"]["rel_ref_c_grad"] += gradient*states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum"]
    states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum"].zero_()

    return states


def calculate_CV_loss(states,gt_data,args,timestep):

    #Calculate simualtion and data ISIs
    #Simulation
    gradient = torch.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets'])), device=torch.device(args['simulation']['device']), dtype=torch.float32)
    loss = gradient.new_zeros(())

    for k in range(args['simulation']['batch_size']):
        for m in range(len(args['simulation']['cell_targets'])):
            isi_holder_sim = []
            isi_holder_data = []
            for n in range(10):
                if len(torch.where(states["neurons"]["Dynamic"]['ron']['spikes_holder'][k,n,m,:] == 1)[0]) > 0:
                    isi_holder_sim.append(torch.diff(torch.where(states["neurons"]["Dynamic"]['ron']['spikes_holder'][k,n,m,:] == 1)[0]))
                
                isi_holder_data.append(torch.diff(torch.where(gt_data[m,n,:] == 1)[0]))

            
            if not isi_holder_sim:
                continue

            sim_isis = torch.cat(isi_holder_sim).float()
            data_isis = torch.cat(isi_holder_data).float()

            #elif torch.mean(torch.cat(isi_holder_sim).to(torch.float32)) == 0 or len(torch.cat(isi_holder_sim).to(torch.float32)) <= 1 or torch.std(torch.cat(isi_holder_sim).to(torch.float32)) == 0: #If mean is zero it will break the grad, or if there is only 1 example it will break the std.
            #    gradient[k,m] = 0
            #else:

            if sim_isis.numel() < 2 or data_isis.numel() < 2:
                continue

            sim_cv = torch.std(torch.cat(isi_holder_sim).to(torch.float32))/torch.mean(torch.cat(isi_holder_sim).to(torch.float32))
            data_cv = torch.std(torch.cat(isi_holder_data).to(torch.float32))/torch.mean(torch.cat(isi_holder_data).to(torch.float32))

            loss = (sim_cv - data_cv)**2
            states["neurons"]["Dynamic"]['ron']['mean_CV_loss'] += loss

            stationary_part_of_the_cv_deriv = 1/(torch.std(torch.cat(isi_holder_sim).to(torch.float32))*(len(torch.cat(isi_holder_sim).to(torch.float32))-1)*torch.mean(torch.cat(isi_holder_sim).to(torch.float32)))

            lamda2 = 2
            gradient[k,m] = 2*(sim_cv - data_cv)*lamda2*stationary_part_of_the_cv_deriv

    return {'loss': loss, 'gradient': gradient}    


def update_grad_CV(states,gradient):
    for k in list(states['synapses']['Static'].keys()): 
        if k.rsplit("_")[1] == "ron":
            states["synapses"]["Learnable"][k]["gSYN_grad"] += gradient*torch.sum(states["synapses"]["Learnable"][k]["gSYN_accum_cv"], dim = 1, keepdim = False)
        else:
            states["synapses"]["Learnable"][k]["gSYN_grad"] += gradient*torch.sum(states["synapses"]["Learnable"][k]["gSYN_accum_cv"], dim = 1, keepdim = False)*states['neurons']['Learnable']['Bk']
        states["synapses"]["Learnable"][k]["gSYN_accum_cv"].zero_()

    states["neurons"]["Learnable"]["ron"]["output_ad_grad"]+= gradient*torch.sum(states["neurons"]["Learnable"]["ron"]["output_ad_accum_cv"], dim = 1, keepdim = False)
    states["neurons"]["Learnable"]["ron"]["output_ad_accum_cv"].zero_()

    states["neurons"]["Learnable"]["STRF_alpha_grad"] += gradient*torch.sum(states["neurons"]["Learnable"]["STRF_alpha_accum_cv"], dim = 1, keepdim = False)
    states["neurons"]["Learnable"]["STRF_alpha_accum_cv"].zero_()

    states["neurons"]["Learnable"]["STRF_gain_grad"] += gradient*torch.sum(states["neurons"]["Learnable"]["STRF_gain_accum_cv"], dim = 1, keepdim = False)
    states["neurons"]["Learnable"]["STRF_gain_accum_cv"].zero_()

    states["neurons"]["Learnable"]["ron"]["abs_ref_grad"] += gradient*torch.sum(states["neurons"]["Learnable"]["ron"]["abs_ref_accum_cv"], dim = 1, keepdim = False)
    states["neurons"]["Learnable"]["ron"]["abs_ref_accum_cv"].zero_()
    states["neurons"]["Learnable"]["ron"]["rel_ref_a_grad"] += gradient*torch.sum(states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum_cv"], dim = 1, keepdim = False)
    states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum_cv"].zero_()
    states["neurons"]["Learnable"]["ron"]["rel_ref_b_grad"] += gradient*torch.sum(states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum_cv"], dim = 1, keepdim = False)
    states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum_cv"].zero_()
    states["neurons"]["Learnable"]["ron"]["rel_ref_c_grad"] += gradient*torch.sum(states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum_cv"], dim = 1, keepdim = False)
    states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum_cv"].zero_()

    return states
