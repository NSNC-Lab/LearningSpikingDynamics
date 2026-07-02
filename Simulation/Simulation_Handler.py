import torch
import time

from Pre_Processing import preprocess_handler
from Simulation import Architecture_Declaration,ode_handler,Eligibility_handler,conditional_handler, Loss_handler, update_handler, Reset_handler

def handle_sim(args, params,gt_data):

    output = run_optimization(args, params, gt_data)
    
    return output

def run_optimization(args, params, gt_data):

    start_time = time.time()
    
    states = Architecture_Declaration.build_network(args, torch.device(args['simulation']['device']), params['params'])

    for epoch in range(args['simulation']['epochs']):
        with torch.no_grad():
            
            pre_processed_object = preprocess_handler.preprocess(args,states,torch.device(args['simulation']['device']))

            for timestep in range(args['simulation']['sim_len']):

                states = ode_handler.run_odes(args, states, pre_processed_object['spks'],timestep)
                states = Eligibility_handler.update_eligibility(args, states, pre_processed_object['onset_offset_rates'],timestep)
                states = conditional_handler.run_conditionals(args, states,timestep)
                states = Loss_handler.handle_loss(args, states, gt_data,timestep)

            states = update_handler.run_adam(states, args, params['lrs'])
            print(f"Epoch {epoch} -- Average SSE: {states['neurons']['Dynamic']['ron']['mean_sse_loss']} -- Average CV: {states['neurons']['Dynamic']['ron']['mean_CV_loss']}")
            states["neurons"]["BookKeeping"].append(states['neurons']['Dynamic']['ron']['mean_sse_loss'])
            if epoch < args['simulation']['epochs'] - 1: #Don't do reset the last epoch because we will save the output
                states = Reset_handler.reset_dyanmics(states, args, torch.device(args['simulation']['device']))

    final_time = time.time() - start_time
    print(f"Simulation completed in {final_time:.2f} seconds.")

    return states
