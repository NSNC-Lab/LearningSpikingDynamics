import numpy as np
from scipy.io import loadmat

def init_params(args):
    params = {}
    lrs = {}

    ###
    #Strf_alpha_mag = 100
    #Strf_gain_mag = 0.02
    output_ad_mag = 0.005
    #on_ron_gSYN_mag = 0.05
    #off_ron_gSYN_mag = 0.05
    on_sonoff_gSYN_mag = 0.05
    off_sonoff_gSYN_mag = 0.05
    sonoff_ron_gSYN_mag = 0.05
    abs_ref_mag = 2
    rel_ref_a_mag = 1
    rel_ref_b_mag = 10
    rel_ref_c_mag = 1

    ###
    #Strf_alpha_min = 1
    #Strf_gain_min = 0.001
    output_ad_min = 0.0001
    #on_ron_gSYN_min = 0.01
    #off_ron_gSYN_min = 0.01
    on_sonoff_gSYN_min = 0.01
    off_sonoff_gSYN_min = 0.01
    sonoff_ron_gSYN_min = 0.01
    abs_ref_min = 0.1
    rel_ref_a_min = 0.01
    rel_ref_b_min = 5
    rel_ref_c_min = 0.2

    #Contrained initialization
    Strf_gain_mag = 0.02
    Strf_gain_min = 0.012

    onsetness = loadmat('C:\\Users\\ipboy\\Documents\\GitHub\\LearningSpikingDynamics\\Data\\Initialization_help\\onsetness.mat')
    pd = loadmat('C:\\Users\\ipboy\\Documents\\GitHub\\LearningSpikingDynamics\\Data\\Initialization_help\\psuedo_densities.mat')

    onset_vals = onsetness['save_onsetness']
    pd_vals = pd['psuedo_densities']

    #Strf_alpha_mag = 40
    #Strf_alpha_min = 20

    #on_ron_gSYN_mag = 0.04
    #on_ron_gSYN_min = 0.03

    #off_ron_gSYN_mag = 0.015
    #off_ron_gSYN_min = 0.01

    params['Strf_alpha'] = (np.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets']))))
    params['on_ron_gSYN'] = (np.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets']))))
    params['off_ron_gSYN'] = (np.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets']))))

    #The following is how I am initializing the necessary parameters so that we get consistant outputs. Previously we tested just intializing everything randomly. Given the netwok is very sensitive to its starting conditions, initializing them to a more narrow area based off the spiking patterns observed allows a higher yeild of batches to have good fit.

    for count, k in enumerate(args['simulation']['cell_targets']):
        zero_indexk = k-1
        Strf_alpha_min = 2*(pd_vals[0][zero_indexk]+4)
        Strf_alpha_mag = 2*(pd_vals[0][zero_indexk]+4)+15
        params['Strf_alpha'][:,count] = np.clip((np.random.rand(args['simulation']['batch_size'],)*(Strf_alpha_mag-Strf_alpha_min)+Strf_alpha_min),0,1000)

        on_ron_gSYN_min = (onset_vals[0][zero_indexk]-0.5)*2*0.02 + 0.015
        on_ron_gSYN_mag = (onset_vals[0][zero_indexk]-0.5)*2*0.02 + 0.025
        params['on_ron_gSYN'][:,count] = np.clip((np.random.rand(args['simulation']['batch_size'],)*(on_ron_gSYN_mag-on_ron_gSYN_min)+on_ron_gSYN_min),0,10)

        off_ron_gSYN_min = ((1-onset_vals[0][zero_indexk])-0.5)*2*0.02 + 0.015
        off_ron_gSYN_mag = ((1-onset_vals[0][zero_indexk])-0.5)*2*0.02 + 0.025
        params['off_ron_gSYN'][:,count] = np.clip((np.random.rand(args['simulation']['batch_size'],)*(off_ron_gSYN_mag-off_ron_gSYN_min)+off_ron_gSYN_min),0,10) #Clamp values to zero if they go below zero. 10 Is the max but it will never get that high

    #params['Strf_alpha'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(Strf_alpha_mag-Strf_alpha_min)+Strf_alpha_min)
    params['Strf_gain'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(Strf_gain_mag-Strf_gain_min)+Strf_gain_min)
    params['output_ad'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(output_ad_mag-output_ad_min)+output_ad_min)
    #params['on_ron_gSYN'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(on_ron_gSYN_mag-on_ron_gSYN_min)+on_ron_gSYN_min)
    #params['off_ron_gSYN'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(off_ron_gSYN_mag-off_ron_gSYN_min)+off_ron_gSYN_min)
    params['on_sonoff_gSYN'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(on_sonoff_gSYN_mag-on_sonoff_gSYN_min)+on_sonoff_gSYN_min)
    params['off_sonoff_gSYN'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(off_sonoff_gSYN_mag-off_sonoff_gSYN_min)+off_sonoff_gSYN_min)
    params['sonoff_ron_gSYN'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(sonoff_ron_gSYN_mag-sonoff_ron_gSYN_min)+sonoff_ron_gSYN_min)
    params['abs_ref'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(abs_ref_mag-abs_ref_min)+abs_ref_min)
    params['rel_ref_a'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(rel_ref_a_mag-rel_ref_a_min)+rel_ref_a_min)
    params['rel_ref_b'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(rel_ref_b_mag-rel_ref_b_min)+rel_ref_b_min)
    params['rel_ref_c'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(rel_ref_c_mag-rel_ref_c_min)+rel_ref_c_min)

    lrs['Strf_alpha'] = Strf_alpha_mag*args['simulation']['lr']
    lrs['Strf_gain'] = Strf_gain_mag*args['simulation']['lr']
    lrs['output_ad'] = output_ad_mag*args['simulation']['lr']
    lrs['on_ron_gSYN'] = on_ron_gSYN_mag*args['simulation']['lr']
    lrs['off_ron_gSYN'] = off_ron_gSYN_mag*args['simulation']['lr']
    lrs['on_sonoff_gSYN'] = on_sonoff_gSYN_mag*args['simulation']['lr']
    lrs['off_sonoff_gSYN'] = off_sonoff_gSYN_mag*args['simulation']['lr']
    lrs['sonoff_ron_gSYN'] = sonoff_ron_gSYN_mag*args['simulation']['lr']
    lrs['abs_ref'] = abs_ref_mag*args['simulation']['lr']
    lrs['rel_ref_a'] = rel_ref_a_mag*args['simulation']['lr']
    lrs['rel_ref_b'] = rel_ref_b_mag*args['simulation']['lr']
    lrs['rel_ref_c'] = rel_ref_c_mag*args['simulation']['lr']

    return {'params': params, 'lrs' : lrs}

