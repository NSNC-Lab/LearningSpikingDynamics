import torch
import time
import numpy as np

from Pre_Processing import preprocess_handler
from Simulation import Architecture_Declaration,ode_handler,Eligibility_handler,conditional_handler, Loss_handler, update_handler, Reset_handler, debug_plot_handler
from Simulation.initialize_from_mat import maybe_restore_adam_from_mat

def handle_sim(args, params,gt_data):

    output = run_optimization(args, params, gt_data)
    
    return output

def run_optimization(args, params, gt_data):

    if bool(args.get("optimization", {}).get("train_refractory", False)):
        raise NotImplementedError(
            "End-to-end refractory training is disabled: the existing CV "
            "multiplier is not d(CV)/d(parameter) for discrete spike times. "
            "Fit refractory recovery with "
            "Archive/Debugging/fit_refractory_hazard.py, or implement an "
            "explicit score-function/renewal objective before enabling it."
        )

    start_time = time.time()

    seed = args.get('simulation', {}).get('seed')
    if seed is not None:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    
    device = torch.device(args['simulation']['device'])
    states = Architecture_Declaration.build_network(args, device, params['params'])
    states = maybe_restore_adam_from_mat(args, states, device)

    for epoch in range(args['simulation']['epochs']):
        with torch.no_grad():
            
            pre_processed_object = preprocess_handler.preprocess(args,states,device)
            debug_plot_handler.plot_input_rates(args, pre_processed_object['onset_offset_rates'], epoch)

            for timestep in range(args['simulation']['sim_len']):

                states = ode_handler.run_odes(args, states, pre_processed_object['spks'],timestep)
                states = Eligibility_handler.update_eligibility(args, states, pre_processed_object['onset_offset_rates'],timestep)
                states = conditional_handler.run_conditionals(args, states,timestep)
                states = Loss_handler.handle_loss(args, states, gt_data,timestep, epoch)

            states = update_handler.run_adam(states, args, params['lrs'],epoch)
            print(f"Epoch {epoch} -- Average SSE: {states['neurons']['Dynamic']['ron']['mean_sse_loss']} -- Average CV: {states['neurons']['Dynamic']['ron']['mean_CV_loss']}")
            firing_rate_hz = states['neurons']['Dynamic']['ron']['spikes_holder'].sum().item()/(args['simulation']['batch_size']*10*len(args['simulation']['cell_targets'])*(args['simulation']['sim_len']*args['simulation']['dt']/1000))
            print(f"Epoch {epoch} -- Average firing rate: {firing_rate_hz:.3f} Hz")
            debug_plot_handler.plot_epoch_summary(args, states, gt_data, epoch)
            states["neurons"]["BookKeeping"].append(states['neurons']['Dynamic']['ron']['mean_sse_loss'])
            states["neurons"].setdefault("FiringRateBookKeeping", []).append(firing_rate_hz)
            cv_value = states['neurons']['Dynamic']['ron']['mean_CV_loss']
            if isinstance(cv_value, torch.Tensor):
                cv_value = float(cv_value.detach().cpu())
            states["neurons"].setdefault("CVBookKeeping", []).append(float(cv_value))
            if epoch < args['simulation']['epochs'] - 1: #Don't do reset the last epoch because we will save the output
                states = Reset_handler.reset_dyanmics(states, args, device)

    final_time = time.time() - start_time
    print(f"Simulation completed in {final_time:.2f} seconds.")

    return states
