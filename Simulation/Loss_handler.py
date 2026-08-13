import torch
import numpy as np

def handle_loss(args, states, gt_data, timestep,epoch):

    if (timestep+1) % args['simulation']['PSTH_granularity'] == 0 and timestep > 0:
        loss = calculate_loss(states, gt_data['psth_holder'], args, timestep,args["simulation"]["PSTH_granularity"],epoch)
        states = update_grad(states, loss['gradient'])

    if timestep == args['simulation']['sim_len'] - 1:
        loss_cv = calculate_CV_loss(states, gt_data['raster_holder'], args, timestep,epoch)
        #states = update_grad_CV(states, loss_cv['gradient'])

        #Firing rate loss

        #loss = calculate_loss(states, gt_data['psth_holder'], args, timestep,29801)
        #states = update_grad(states, loss['gradient']*100)




    return states

def calculate_loss(states,gt_data,args,timestep, granularity,epoch):


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
    gradient = 2*(sim_psth - gt_data[None,:,int(timestep/args["simulation"]["PSTH_granularity"])-1] - 0.5)
    return {'loss': loss, 'gradient': gradient}

def calculate_CV_loss(states,gt_data,args,timestep,epoch):

    #Calculate simualtion and data ISIs
    #Simulation
    gradient = torch.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets'])), device=torch.device(args['simulation']['device']), dtype=torch.float32)
    gradient_a = torch.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets'])), device=torch.device(args['simulation']['device']), dtype=torch.float32)
    gradient_b = torch.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets'])), device=torch.device(args['simulation']['device']), dtype=torch.float32)
    gradient_c = torch.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets'])), device=torch.device(args['simulation']['device']), dtype=torch.float32)

    ds_dtheta = states['neurons']["Running_grads"]['ron']["ds_dtheta"]
    ds_dtheta_a = states['neurons']["Running_grads"]['ron']["ds_dtheta_a"]
    ds_dtheta_b = states['neurons']["Running_grads"]['ron']["ds_dtheta_b"]
    ds_dtheta_c = states['neurons']["Running_grads"]['ron']["ds_dtheta_c"]
    ds_dtheta_identity = states['neurons']["Running_grads"]['ron']["ds_dtheta_identity"]

    if len(ds_dtheta) == 0:
        loss = 100000
        states["neurons"]["Dynamic"]['ron']['mean_CV_loss'] += loss
        states["neurons"]["Dynamic"]['ron']['all_CV_loss'][:,:,epoch] = loss
    else:
        ds_all = torch.cat(ds_dtheta)
        ds_all_a = torch.cat(ds_dtheta_a)
        ds_all_b = torch.cat(ds_dtheta_b)
        ds_all_c = torch.cat(ds_dtheta_c)

        batch_all = torch.cat([chunk[0] for chunk in ds_dtheta_identity])
        trial_all = torch.cat([chunk[1] for chunk in ds_dtheta_identity])
        cell_all  = torch.cat([chunk[2] for chunk in ds_dtheta_identity])

        analysis_end = args['simulation']['sim_len'] - 5

        sim_raster  = states["neurons"]["Dynamic"]['ron']['spikes_holder'][..., :analysis_end]
        data_raster = gt_data[..., :analysis_end]

        for k in range(args['simulation']['batch_size']):
            for m in range(len(args['simulation']['cell_targets'])):
                isi_holder_sim = []
                isi_holder_data = []
                isi_holder_grad = []
                isi_holder_grad_a = []
                isi_holder_grad_b = []
                isi_holder_grad_c = []
                for n in range(10):
                    if len(torch.where(sim_raster[k,n,m,:] == 1)[0]) > 0:
                        isi_holder_sim.append(torch.diff(torch.where(sim_raster[k,n,m,:] == 1)[0]))

                        #Extract out the locations of the necessary dsi/dthetas
                        indexes = torch.where((batch_all == k) & (trial_all == n) & (cell_all == m))

                        ds_stream = ds_all[indexes]
                        ds_stream_a = ds_all_a[indexes]
                        ds_stream_b = ds_all_b[indexes]
                        ds_stream_c = ds_all_c[indexes]
                        d_isi_stream = torch.diff(ds_stream) / args['simulation']['dt']
                        d_isi_stream_a = torch.diff(ds_stream_a) / args['simulation']['dt']
                        d_isi_stream_b = torch.diff(ds_stream_b) / args['simulation']['dt']
                        d_isi_stream_c = torch.diff(ds_stream_c) / args['simulation']['dt']
                        isi_holder_grad.append(d_isi_stream)
                        isi_holder_grad_a.append(d_isi_stream_a)
                        isi_holder_grad_b.append(d_isi_stream_b)
                        isi_holder_grad_c.append(d_isi_stream_c)

                    isi_holder_data.append(torch.diff(torch.where(data_raster[m,n,:] == 1)[0]))

                if len(isi_holder_sim) == 0:
                    gradient[k,m] = 0
                    data_cv = torch.std(torch.cat(isi_holder_data).to(torch.float32))/torch.mean(torch.cat(isi_holder_data).to(torch.float32))
                    loss = data_cv**2
                    states["neurons"]["Dynamic"]['ron']['all_CV_loss'][k,m,epoch] += loss
                elif torch.mean(torch.cat(isi_holder_sim).to(torch.float32)) == 0 or len(torch.cat(isi_holder_sim).to(torch.float32)) <= 1: #If mean is zero it will break the grad, or if there is only 1 example it will break the std.
                    gradient[k,m] = 0
                    data_cv = torch.std(torch.cat(isi_holder_data).to(torch.float32))/torch.mean(torch.cat(isi_holder_data).to(torch.float32))
                    loss = data_cv**2
                    states["neurons"]["Dynamic"]['ron']['all_CV_loss'][k,m,epoch] += loss
                else:
                    sim_cv = torch.std(torch.cat(isi_holder_sim).to(torch.float32))/torch.mean(torch.cat(isi_holder_sim).to(torch.float32))
                    data_cv = torch.std(torch.cat(isi_holder_data).to(torch.float32))/torch.mean(torch.cat(isi_holder_data).to(torch.float32))

                    mean_cv_val = torch.mean(torch.cat(isi_holder_sim).to(torch.float32))
                    std_cv_val =  torch.std(torch.cat(isi_holder_sim).to(torch.float32))

                    if std_cv_val == 0:
                        gradient[k,m] = 0
                        loss = data_cv**2
                        states["neurons"]["Dynamic"]['ron']['all_CV_loss'][k,m,epoch] += loss
                    else:
                        dCV_ddeltai = (torch.cat(isi_holder_sim).to(torch.float32) - mean_cv_val)/((len(torch.cat(isi_holder_sim).to(torch.float32))-1) * std_cv_val * mean_cv_val) - std_cv_val/(len(torch.cat(isi_holder_sim).to(torch.float32))*mean_cv_val**2)

                        grad_vals = torch.cat(isi_holder_grad).to(torch.float32)
                        grad_vals_a = torch.cat(isi_holder_grad_a).to(torch.float32)
                        grad_vals_b = torch.cat(isi_holder_grad_b).to(torch.float32)
                        grad_vals_c = torch.cat(isi_holder_grad_c).to(torch.float32)

                        dCV_dtheta = torch.sum(dCV_ddeltai * grad_vals)
                        dCV_dtheta_a = torch.sum(dCV_ddeltai * grad_vals_a)
                        dCV_dtheta_b = torch.sum(dCV_ddeltai * grad_vals_b)
                        dCV_dtheta_c = torch.sum(dCV_ddeltai * grad_vals_c)

                        loss = (sim_cv - data_cv)**2
                        states["neurons"]["Dynamic"]['ron']['mean_CV_loss'] += loss
                        states["neurons"]["Dynamic"]['ron']['all_CV_loss'][k,m,epoch] += loss

                        gradient[k,m] = 2*(sim_cv - data_cv)*dCV_dtheta
                        gradient_a[k,m] = 2*(sim_cv - data_cv)*dCV_dtheta_a
                        gradient_b[k,m] = 2*(sim_cv - data_cv)*dCV_dtheta_b
                        gradient_c[k,m] = 2*(sim_cv - data_cv)*dCV_dtheta_c

    states["neurons"]["Learnable"]["ron"]["abs_ref_grad"] += gradient
    states["neurons"]["Learnable"]["ron"]["rel_ref_a_grad"] += gradient_a
    states["neurons"]["Learnable"]["ron"]["rel_ref_b_grad"] += gradient_b
    states["neurons"]["Learnable"]["ron"]["rel_ref_c_grad"] += gradient_c

    #print(gradient_a)
        
    return {'loss': loss, 'gradient': gradient}    

def update_grad(states,gradient):
    for k in list(states['synapses']['Static'].keys()): 
        if k.rsplit("_")[1] == "ron":
            states["synapses"]["Learnable"][k]["gSYN_grad"] += gradient*states["synapses"]["Learnable"][k]["gSYN_accum"]
        else:
            states["synapses"]["Learnable"][k]["gSYN_grad"] += gradient*states["synapses"]["Learnable"][k]["gSYN_accum"]
        states["synapses"]["Learnable"][k]["gSYN_accum"].zero_()

    states["neurons"]["Learnable"]["ron"]["output_ad_grad"]+= gradient*states["neurons"]["Learnable"]["ron"]["output_ad_accum"]
    states["neurons"]["Learnable"]["ron"]["output_ad_accum"].zero_()

    states["neurons"]["Learnable"]["STRF_alpha_grad"] += gradient*states["neurons"]["Learnable"]["STRF_alpha_accum"]
    states["neurons"]["Learnable"]["STRF_alpha_accum"].zero_()

    states["neurons"]["Learnable"]["STRF_gain_grad"] += gradient*states["neurons"]["Learnable"]["STRF_gain_accum"]
    states["neurons"]["Learnable"]["STRF_gain_accum"].zero_()

    return states

# def update_grad_CV(states,gradient):

#     states["neurons"]["Learnable"]["ron"]["abs_ref_grad"] += gradient
#     states["neurons"]["Learnable"]["ron"]["rel_ref_a_grad"] += gradient*states["neurons"]["Learnable"]["ron"]["rel_ref_a_accum"]
#     states["neurons"]["Learnable"]["ron"]["rel_ref_b_grad"] += gradient*states["neurons"]["Learnable"]["ron"]["rel_ref_b_accum"]
#     states["neurons"]["Learnable"]["ron"]["rel_ref_c_grad"] += gradient*states["neurons"]["Learnable"]["ron"]["rel_ref_c_accum"]

#     return states
