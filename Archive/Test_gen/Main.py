#-- This file contains the main execution logic for the spiking dynamics simulation
# 1. Preprocessing
# 2. Simulation and gradient accumulation
# 3. Gradient Updates

from Simulation import Simulation_Handler,Parameter_initialization,output_handler
from Data import data_handler
import yaml
import numpy as np
from pathlib import Path

def main():

    #Load in simulation configureation
    args = yaml.safe_load(Path(__file__).with_name('simulation_config.yaml').read_text())

    #Initialize learnable parameters
    params = Parameter_initialization.init_params(args)

    #Load gt data
    gt_data = data_handler.load_gt_data(args)
    
    #Run the simulation
    output = Simulation_Handler.handle_sim(args,params,gt_data)

    #Save output
    output_handler.save_output(output, checkpoint=True)

    return output

if __name__ == "__main__":
    main()
