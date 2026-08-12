from Pre_Processing import pre_cortical_handler, pre_cortical_handler_legacy_match
from Pre_Processing import spiking_handler

def preprocess(args,params,device):

    #Calculate STRF
    onset_offset_rates = get_pre_cortical_handler(args).create_input_fr(args,params, device)
    spks = spiking_handler.generate_spike_trains(onset_offset_rates,args,device)
    
    return {'spks': spks, 'onset_offset_rates': onset_offset_rates}

def get_pre_cortical_handler(args):
    handler_name = args.get("preprocessing", {}).get("pre_cortical_handler", "current")
    if handler_name == "legacy_match":
        return pre_cortical_handler_legacy_match
    return pre_cortical_handler
