from Pre_Processing import pre_cortical_handler
from Pre_Processing import spiking_handler

def preprocess(args,params,device):

    #Calculate STRF
    onset_offset_rates = pre_cortical_handler.create_input_fr(args,params, device)
    spks = spiking_handler.generate_spike_trains(onset_offset_rates,args,device)
    
    return {'spks': spks, 'onset_offset_rates': onset_offset_rates}