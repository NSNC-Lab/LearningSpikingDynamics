import torch
import numpy as np

from Simulation import debug_plot_handler

def handle_loss(args, states, gt_data, timestep, epoch=0):

    # Close a bin only after all G samples (including the current one) have
    # been simulated.  This keeps spike counts and eligibility samples in the
    # same [t-G+1, t] interval.
    if (timestep + 1) % args['simulation']['PSTH_granularity'] == 0:
        loss = calculate_loss(states, gt_data['psth_holder'], args, timestep,args["simulation"]["PSTH_granularity"])
        states = update_grad(states, loss['gradient'])
        debug_plot_handler.plot_loss_update(args, states, gt_data, timestep + 1, epoch)

    if timestep == args['simulation']['sim_len'] - 1:
        loss_cv = calculate_CV_loss(states, gt_data['raster_holder'], args, timestep)
        if bool(args.get("optimization", {}).get("train_refractory", False)):
            states = update_grad_CV(states, loss_cv['gradient'])

        #Firing rate loss

        #loss = calculate_loss(states, gt_data['psth_holder'], args, timestep,29801)
        #states = update_grad(states, loss['gradient']*100)




    return states

def calculate_loss(states,gt_data,args,timestep, granularity):

    #Caluclate simulation PSTHs
    #Trim 
    bin_end = timestep + 1
    bin_start = bin_end - granularity
    holder = states["neurons"]["Dynamic"]['ron']['spikes_holder'][:,:,:,bin_start:bin_end]
    #Sums across trial dim
    holder = torch.sum(holder, dim = 1,keepdim = False)
    #Reshape in order to sum across bins
    #holder = holder.reshape((args['simulation']['batch_size'],len(args['simulation']['cell_targets']),int(np.round(args["simulation"]["sim_len"]/args["simulation"]["PSTH_granularity"])),args["simulation"]["PSTH_granularity"]))
    sim_psth = torch.sum(holder, dim = -1,keepdim = False)

    #Calculate Loss and gradient
    #Average SSE
    target_index = int(bin_end/granularity) - 1
    target = gt_data[None,:,target_index]
    loss = (sim_psth - target)**2
    states["neurons"]["Dynamic"]['ron']['mean_sse_loss'] += torch.mean(loss.flatten()).cpu()

    # A sampled Poisson-like count MSE has a 0.5-count downward optimum bias:
    # E[(S-Y)^2] = Var(S) + (E[S]-Y)^2.  Subtracting 1 from the pathwise
    # multiplier (equivalently targeting Y+0.5) removes the leading-order
    # count bias.  Keep the reported loss against the unshifted data target.
    count_bias_correction = float(args.get("loss", {}).get("psth_count_bias_correction", 0.0))
    gradient = 2*(sim_psth - target - count_bias_correction)
    return {'loss': loss, 'gradient': gradient}

def calculate_CV_loss(states,gt_data,args,timestep):

    #Calculate simualtion and data ISIs
    #Simulation
    gradient = torch.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets'])), device=torch.device(args['simulation']['device']), dtype=torch.float32)
    loss = torch.tensor(0.0, device=torch.device(args['simulation']['device']))

    for k in range(args['simulation']['batch_size']):
        for m in range(len(args['simulation']['cell_targets'])):
            isi_holder_sim = []
            isi_holder_data = []
            for n in range(10):
                sim_spike_times = torch.where(states["neurons"]["Dynamic"]['ron']['spikes_holder'][k,n,m,:] == 1)[0]
                data_spike_times = torch.where(gt_data[m,n,:] == 1)[0]
                if len(sim_spike_times) > 1:
                    isi_holder_sim.append(torch.diff(sim_spike_times))
                if len(data_spike_times) > 1:
                    isi_holder_data.append(torch.diff(data_spike_times))

            if not isi_holder_sim or not isi_holder_data:
                gradient[k,m] = 0
                continue

            sim_isis = torch.cat(isi_holder_sim).to(torch.float32)
            data_isis = torch.cat(isi_holder_data).to(torch.float32)
            if sim_isis.numel() <= 1 or data_isis.numel() <= 1 or torch.mean(sim_isis) == 0 or torch.mean(data_isis) == 0:
                gradient[k,m] = 0
            else:
                sim_cv = torch.std(sim_isis)/torch.mean(sim_isis)
                data_cv = torch.std(data_isis)/torch.mean(data_isis)

                loss = (sim_cv - data_cv)**2
                states["neurons"]["Dynamic"]['ron']['mean_CV_loss'] += loss
                
                gradient[k,m] = 2*(sim_cv - data_cv)

    return {'loss': loss, 'gradient': gradient}    

def update_grad(states,gradient):
    for k in list(states['synapses']['Static'].keys()): 
        # Every accumulator is now d(output spike surrogate)/d(the named
        # conductance), including the full on/off -> sonoff -> ron path.
        states["synapses"]["Learnable"][k]["gSYN_grad"] += gradient*states["synapses"]["Learnable"][k]["gSYN_accum"]
        states["synapses"]["Learnable"][k]["gSYN_accum"].zero_()

    states["neurons"]["Learnable"]["ron"]["output_ad_grad"]+= gradient*states["neurons"]["Learnable"]["ron"]["output_ad_accum"]
    states["neurons"]["Learnable"]["ron"]["output_ad_accum"].zero_()

    states["neurons"]["Learnable"]["STRF_alpha_grad"] += gradient*states["neurons"]["Learnable"]["STRF_alpha_accum"]
    states["neurons"]["Learnable"]["STRF_alpha_accum"].zero_()

    states["neurons"]["Learnable"]["STRF_gain_grad"] += gradient*states["neurons"]["Learnable"]["STRF_gain_accum"]
    states["neurons"]["Learnable"]["STRF_gain_accum"].zero_()

    return states

def update_grad_CV(states,gradient):

    states["neurons"]["Learnable"]["ron"]["abs_ref_grad"] += gradient*states["neurons"]["Learnable"]["ron"]["abs_ref_accum"]
    states["neurons"]["Learnable"]["ron"]["rel_ref_a_grad"] += gradient*states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum"]
    states["neurons"]["Learnable"]["ron"]["rel_ref_b_grad"] += gradient*states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum"]
    states["neurons"]["Learnable"]["ron"]["rel_ref_c_grad"] += gradient*states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum"]

    return states
