import torch
import time

from Pre_Processing import preprocess_handler
from Simulation import Architecture_Declaration,ode_handler,Eligibility_handler,conditional_handler, Loss_handler, update_handler, Reset_handler, output_handler
from Simulation.initialize_from_mat import maybe_restore_adam_from_mat

from pathlib import Path

def handle_sim(args, params,gt_data):

    output = run_optimization(args, params, gt_data)
    
    return output

def run_optimization(args, params, gt_data):

    start_time = time.time()
    
    device = torch.device(args['simulation']['device'])
    states = Architecture_Declaration.build_network(args, device, params['params'])
    states = maybe_restore_adam_from_mat(args, states, device)
    start_epoch = states["neurons"]["Adam"]["t"]
    raster_dir = Path(__file__).resolve().parents[1] / "Epoch_Rasters"
    raster_dir.mkdir(exist_ok=True)

    for epoch in range(args['simulation']['epochs']):
        with torch.no_grad():
            
            pre_processed_object = preprocess_handler.preprocess(args,states,torch.device(args['simulation']['device']))

            #print('rel_ref_amplitude')
            #print(states["neurons"]["Learnable"]["ron"]["rel_ref_c"])

            for timestep in range(args['simulation']['sim_len']):

                states = ode_handler.run_odes(args, states, pre_processed_object['spks'], pre_processed_object['onset_offset_rates'], timestep)
                states = conditional_handler.run_conditionals(args, states,timestep)
                states = Eligibility_handler.update_eligibility(args, states, pre_processed_object['onset_offset_rates'],timestep)
                states = Loss_handler.handle_loss(args, states, gt_data,timestep,epoch)

            states = update_handler.run_adam(states, args, params['lrs'],epoch)
            print(f"Epoch {start_epoch+epoch} -- Average SSE: {states['neurons']['Dynamic']['ron']['mean_sse_loss']} -- Average CV: {states['neurons']['Dynamic']['ron']['mean_CV_loss']}")
            print(f"Epoch {start_epoch+epoch} -- Average firing rate: {states['neurons']['Dynamic']['ron']['spikes_holder'].sum().item()/(args['simulation']['batch_size']*10*len(args['simulation']['cell_targets'])*(args['simulation']['sim_len']*args['simulation']['dt']/1000)):.3f} Hz")
            states["neurons"]["BookKeeping"].append(states['neurons']['Dynamic']['ron']['mean_sse_loss'])
            cv_value = states['neurons']['Dynamic']['ron']['mean_CV_loss']
            if isinstance(cv_value, torch.Tensor):
                cv_value = float(cv_value.detach().cpu())
            states["neurons"].setdefault("CVBookKeeping", []).append(float(cv_value))
            output_handler.save_output(states,raster_dir / f"rasters_epoch_{states['neurons']['Adam']['t']:03d}.mat",epoch+1)
            if epoch < args['simulation']['epochs'] - 1: #Don't do reset the last epoch because we will save the output
                states = Reset_handler.reset_dyanmics(states, args, torch.device(args['simulation']['device']))

    final_time = time.time() - start_time
    print(f"Simulation completed in {final_time:.2f} seconds.")

    return states
