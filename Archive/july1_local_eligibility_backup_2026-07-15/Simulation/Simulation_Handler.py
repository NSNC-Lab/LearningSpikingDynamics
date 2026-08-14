import torch
import time

from Pre_Processing import preprocess_handler
from Simulation import Architecture_Declaration,ode_handler,Eligibility_handler,conditional_handler, Loss_handler, update_handler, Reset_handler, parameter_saving

from pathlib import Path

def handle_sim(args, params,gt_data):

    output = run_optimization(args, params, gt_data)
    
    return output

def run_optimization(args, params, gt_data):

    start_time = time.time()
    
    states = Architecture_Declaration.build_network(args, torch.device(args['simulation']['device']), params['params'])
    checkpoint = args["simulation"].get("checkpoint")
    if checkpoint:
        parameter_saving.restore(states,checkpoint)
    parameter_saving.initialize(states,args['simulation']['epochs'])
    start_epoch = states["neurons"]["Adam"]["t"]
    raster_dir = Path(__file__).resolve().parents[1] / "Epoch_Rasters_Eprop"
    raster_dir.mkdir(exist_ok=True)

    for epoch in range(args['simulation']['epochs']):
        with torch.no_grad():
            
            pre_processed_object = preprocess_handler.preprocess(args,states,torch.device(args['simulation']['device']))

            for timestep in range(args['simulation']['sim_len']):

                states = ode_handler.run_odes(args, states, pre_processed_object['spks'],timestep)
                states = conditional_handler.run_conditionals(args, states,timestep)
                states = Eligibility_handler.update_eligibility(args, states, pre_processed_object['onset_offset_rates'],timestep)
                states = Loss_handler.handle_loss(args, states, gt_data,timestep)

            parameter_saving.record(states,epoch)
            states = update_handler.run_adam(states, args, params['lrs'])
            print(f"Epoch {start_epoch+epoch} -- Average SSE: {states['neurons']['Dynamic']['ron']['mean_sse_loss']} -- Average CV: {states['neurons']['Dynamic']['ron']['mean_CV_loss']}")
            print(f"Epoch {start_epoch+epoch} -- Average firing rate: {states['neurons']['Dynamic']['ron']['spikes_holder'].sum().item()/(args['simulation']['batch_size']*10*len(args['simulation']['cell_targets'])*(args['simulation']['sim_len']*args['simulation']['dt']/1000)):.3f} Hz")
            states["neurons"]["BookKeeping"].append(states['neurons']['Dynamic']['ron']['mean_sse_loss'])
            parameter_saving.save(states,raster_dir / f"rasters_epoch_{start_epoch+epoch+1:03d}.mat",checkpoint=True)
            if epoch < args['simulation']['epochs'] - 1: #Don't do reset the last epoch because we will save the output
                states = Reset_handler.reset_dyanmics(states, args, torch.device(args['simulation']['device']))

    final_time = time.time() - start_time
    print(f"Simulation completed in {final_time:.2f} seconds.")

    return states
