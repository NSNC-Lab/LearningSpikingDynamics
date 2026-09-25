import numpy as np
from scipy.io import loadmat

try:
    from Simulation.initialize_from_mat import maybe_replace_initialization_from_mat
except ModuleNotFoundError:
    from initialize_from_mat import maybe_replace_initialization_from_mat

def init_params(args):
    params = {}
    lrs = {}


    #Going to try to cover things by initilizing in a the following regiemes. 3x per regieme to taling batch size = 21
    # Sharp Onset
    # Sharp Offset
    # Sharp Both
    # Broader Onset
    # Broader Offset
    # Broader Both
    # Tonic lower Firing

    kz = 1

    #Sharp Onset
    params['Strf_alpha'] = np.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets'])))
    params['Strf_alpha'][0:kz,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*5+2
    params['Strf_gain'] = np.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets'])))
    params['Strf_gain'][0:kz,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.03+0.03
    params['output_ad'] = np.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets'])))
    params['output_ad'][0:kz,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.001+0.001
    params['on_ron_gSYN'] = np.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets'])))
    params['on_ron_gSYN'][0:kz,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.04+0.04
    params['off_ron_gSYN'] = np.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets'])))
    params['off_ron_gSYN'][0:kz,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0+0
    params['on_sonoff_gSYN'] = np.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets'])))
    params['on_sonoff_gSYN'][0:kz,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.03+0.03
    params['off_sonoff_gSYN'] = np.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets'])))
    params['off_sonoff_gSYN'][0:kz,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.03+0.03
    params['sonoff_ron_gSYN'] = np.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets'])))
    params['sonoff_ron_gSYN'][0:kz,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.03+0.03
    params['abs_ref'] = np.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets'])))
    params['abs_ref'][0:kz,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*2+0.1
    params['rel_ref_a'] = np.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets'])))
    params['rel_ref_a'][0:kz,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*1+0.01
    params['rel_ref_b'] = np.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets'])))
    params['rel_ref_b'][0:kz,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*10+5
    params['rel_ref_c'] = np.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets'])))
    params['rel_ref_c'][0:kz,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*1+0.5

    #Sharp Offset
    params['Strf_alpha'][kz:kz*2,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*5+2
    params['Strf_gain'][kz:kz*2,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.03+0.03
    params['output_ad'][kz:kz*2,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.001+0.001
    params['on_ron_gSYN'][kz:kz*2,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0+0
    params['off_ron_gSYN'][kz:kz*2,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.04+0.04
    params['on_sonoff_gSYN'][kz:kz*2,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.00+0.00
    params['off_sonoff_gSYN'][kz:kz*2,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.00+0.00
    params['sonoff_ron_gSYN'][kz:kz*2,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.00+0.00
    params['abs_ref'][kz:kz*2,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0+0.1
    params['rel_ref_a'][kz:kz*2,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*1+0.01
    params['rel_ref_b'][kz:kz*2,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*10+5
    params['rel_ref_c'][kz:kz*2,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*1+0.5

    #Sharp Both
    params['Strf_alpha'][kz*2:kz*3,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*5+2
    params['Strf_gain'][kz*2:kz*3,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.03+0.03
    params['output_ad'][kz*2:kz*3,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.001+0.001
    params['on_ron_gSYN'][kz*2:kz*3,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.04+0.04
    params['off_ron_gSYN'][kz*2:kz*3,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.04+0.04
    params['on_sonoff_gSYN'][kz*2:kz*3,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.03+0.03
    params['off_sonoff_gSYN'][kz*2:kz*3,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.03+0.03
    params['sonoff_ron_gSYN'][kz*2:kz*3,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.03+0.03
    params['abs_ref'][kz*2:kz*3,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*2+0.1
    params['rel_ref_a'][kz*2:kz*3,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*1+0.01
    params['rel_ref_b'][kz*2:kz*3,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*10+5
    params['rel_ref_c'][kz*2:kz*3,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*1+0.5


    #Broader Onset
    params['Strf_alpha'][kz*3:kz*4,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*30+10
    params['Strf_gain'][kz*3:kz*4,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.03+0.03
    params['output_ad'][kz*3:kz*4,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.001+0.001
    params['on_ron_gSYN'][kz*3:kz*4,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.04+0.04
    params['off_ron_gSYN'][kz*3:kz*4,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0+0
    params['on_sonoff_gSYN'][kz*3:kz*4,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.03+0.03
    params['off_sonoff_gSYN'][kz*3:kz*4,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.03+0.03
    params['sonoff_ron_gSYN'][kz*3:kz*4,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.03+0.03
    params['abs_ref'][kz*3:kz*4,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*2+0.1
    params['rel_ref_a'][kz*3:kz*4,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*1+0.01
    params['rel_ref_b'][kz*3:kz*4,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*10+5
    params['rel_ref_c'][kz*3:kz*4,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*1+0.5

    #Broader Offset
    params['Strf_alpha'][kz*4:kz*5,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*30+10
    params['Strf_gain'][kz*4:kz*5,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.03+0.03
    params['output_ad'][kz*4:kz*5,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.001+0.001
    params['on_ron_gSYN'][kz*4:kz*5,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0+0
    params['off_ron_gSYN'][kz*4:kz*5,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.04+0.04
    params['on_sonoff_gSYN'][kz*4:kz*5,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.03+0.03
    params['off_sonoff_gSYN'][kz*4:kz*5,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.03+0.03
    params['sonoff_ron_gSYN'][kz*4:kz*5,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.03+0.03
    params['abs_ref'][kz*4:kz*5,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*2+0.1
    params['rel_ref_a'][kz*4:kz*5,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*1+0.01
    params['rel_ref_b'][kz*4:kz*5,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*10+5
    params['rel_ref_c'][kz*4:kz*5,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*1+0.5

    #Broader Both
    params['Strf_alpha'][kz*5:kz*6,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*30+10
    params['Strf_gain'][kz*5:kz*6,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.03+0.03
    params['output_ad'][kz*5:kz*6,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.001+0.001
    params['on_ron_gSYN'][kz*5:kz*6,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.04+0.04
    params['off_ron_gSYN'][kz*5:kz*6,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.04+0.04
    params['on_sonoff_gSYN'][kz*5:kz*6,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.03+0.03
    params['off_sonoff_gSYN'][kz*5:kz*6,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.03+0.03
    params['sonoff_ron_gSYN'][kz*5:kz*6,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.03+0.03
    params['abs_ref'][kz*5:kz*6,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*2+0.1
    params['rel_ref_a'][kz*5:kz*6,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*1+0.01
    params['rel_ref_b'][kz*5:kz*6,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*10+5
    params['rel_ref_c'][kz*5:kz*6,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*1+0.5

    #Tonic lower FR
    params['Strf_alpha'][kz*6:kz*7,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*30+10
    params['Strf_gain'][kz*6:kz*7,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.005+0.001
    params['output_ad'][kz*6:kz*7,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.001+0.001
    params['on_ron_gSYN'][kz*6:kz*7,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.02+0.02
    params['off_ron_gSYN'][kz*6:kz*7,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.02+0.02
    params['on_sonoff_gSYN'][kz*6:kz*7,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.01+0.01
    params['off_sonoff_gSYN'][kz*6:kz*7,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.01+0.01
    params['sonoff_ron_gSYN'][kz*6:kz*7,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*0.01+0.01
    params['abs_ref'][kz*6:kz*7,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*2+0.1
    params['rel_ref_a'][kz*6:kz*7,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*1+0.01
    params['rel_ref_b'][kz*6:kz*7,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*10+5
    params['rel_ref_c'][kz*6:kz*7,:] = np.random.rand(kz,len(args['simulation']['cell_targets']))*1+0.5


    # ###
    # #Strf_alpha_mag = 100
    # #Strf_gain_mag = 0.02
    # output_ad_mag = 0.005
    # #on_ron_gSYN_mag = 0.05
    # #off_ron_gSYN_mag = 0.05
    # on_sonoff_gSYN_mag = 0.05
    # off_sonoff_gSYN_mag = 0.05
    # sonoff_ron_gSYN_mag = 0.05
    # abs_ref_mag = 2
    # rel_ref_a_mag = 1
    # rel_ref_b_mag = 10
    # rel_ref_c_mag = 1

    # ###
    # #Strf_alpha_min = 1
    # #Strf_gain_min = 0.001
    # output_ad_min = 0.0001
    # #on_ron_gSYN_min = 0.01
    # #off_ron_gSYN_min = 0.01
    # on_sonoff_gSYN_min = 0.01
    # off_sonoff_gSYN_min = 0.01
    # sonoff_ron_gSYN_min = 0.01
    # abs_ref_min = 0.1
    # rel_ref_a_min = 0.01
    # rel_ref_b_min = 5
    # rel_ref_c_min = 0.2

    # #Contrained initialization
    # Strf_gain_mag = 0.02
    # Strf_gain_min = 0.012

    # onsetness = loadmat('C:\\Users\\ipboy\\Documents\\GitHub\\LearningSpikingDynamics\\Data\\Initialization_help\\onsetness.mat')
    # pd = loadmat('C:\\Users\\ipboy\\Documents\\GitHub\\LearningSpikingDynamics\\Data\\Initialization_help\\psuedo_densities.mat')

    # #onset_vals = onsetness['save_onsetness']
    # #pd_vals = pd['psuedo_densities']
    # onset_vals = [[1]]
    # pd_vals = [[0]]


    # #Strf_alpha_mag = 40
    # #Strf_alpha_min = 20

    # #on_ron_gSYN_mag = 0.04
    # #on_ron_gSYN_min = 0.03

    # #off_ron_gSYN_mag = 0.015
    # #off_ron_gSYN_min = 0.01

    # params['Strf_alpha'] = (np.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets']))))
    # params['on_ron_gSYN'] = (np.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets']))))
    # params['off_ron_gSYN'] = (np.zeros((args['simulation']['batch_size'],len(args['simulation']['cell_targets']))))

    # #The following is how I am initializing the necessary parameters so that we get consistant outputs. Previously we tested just intializing everything randomly. Given the netwok is very sensitive to its starting conditions, initializing them to a more narrow area based off the spiking patterns observed allows a higher yeild of batches to have good fit.

    # for count, k in enumerate(args['simulation']['cell_targets']):
    #     #zero_indexk = k-1
    #     zero_indexk = count
    #     Strf_alpha_min = 2*(pd_vals[0][zero_indexk]+2)
    #     Strf_alpha_mag = 2*(pd_vals[0][zero_indexk]+2)+5
    #     params['Strf_alpha'][:,count] = np.clip((np.random.rand(args['simulation']['batch_size'],)*(Strf_alpha_mag-Strf_alpha_min)+Strf_alpha_min),0,1000)

    #     on_ron_gSYN_min = (onset_vals[0][zero_indexk]-0.5)*2*0.02 + 0.015
    #     on_ron_gSYN_mag = (onset_vals[0][zero_indexk]-0.5)*2*0.02 + 0.025
    #     params['on_ron_gSYN'][:,count] = np.clip((np.random.rand(args['simulation']['batch_size'],)*(on_ron_gSYN_mag-on_ron_gSYN_min)+on_ron_gSYN_min),0,10)

    #     off_ron_gSYN_min = ((1-onset_vals[0][zero_indexk])-0.5)*2*0.02 + 0.015
    #     off_ron_gSYN_mag = ((1-onset_vals[0][zero_indexk])-0.5)*2*0.02 + 0.025
    #     params['off_ron_gSYN'][:,count] = np.clip((np.random.rand(args['simulation']['batch_size'],)*(off_ron_gSYN_mag-off_ron_gSYN_min)+off_ron_gSYN_min),0,10) #Clamp values to zero if they go below zero. 10 Is the max but it will never get that high

    # #params['Strf_alpha'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(Strf_alpha_mag-Strf_alpha_min)+Strf_alpha_min)
    # params['Strf_gain'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(Strf_gain_mag-Strf_gain_min)+Strf_gain_min)
    # params['output_ad'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(output_ad_mag-output_ad_min)+output_ad_min)
    # #params['on_ron_gSYN'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(on_ron_gSYN_mag-on_ron_gSYN_min)+on_ron_gSYN_min)
    # #params['off_ron_gSYN'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(off_ron_gSYN_mag-off_ron_gSYN_min)+off_ron_gSYN_min)
    # params['on_sonoff_gSYN'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(on_sonoff_gSYN_mag-on_sonoff_gSYN_min)+on_sonoff_gSYN_min)
    # params['off_sonoff_gSYN'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(off_sonoff_gSYN_mag-off_sonoff_gSYN_min)+off_sonoff_gSYN_min)
    # params['sonoff_ron_gSYN'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(sonoff_ron_gSYN_mag-sonoff_ron_gSYN_min)+sonoff_ron_gSYN_min)
    # params['abs_ref'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(abs_ref_mag-abs_ref_min)+abs_ref_min)
    # params['rel_ref_a'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(rel_ref_a_mag-rel_ref_a_min)+rel_ref_a_min)
    # params['rel_ref_b'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(rel_ref_b_mag-rel_ref_b_min)+rel_ref_b_min)
    # params['rel_ref_c'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(rel_ref_c_mag-rel_ref_c_min)+rel_ref_c_min)

    # # saved = loadmat('C:\\Users\\ipboy\\Documents\\GitHub\\LearningSpikingDynamics\\186_10',simplify_cells=True)

    # params = maybe_replace_initialization_from_mat(args, params)

    # # #Start things by replacing all of the parameters
    # # params['Strf_alpha'] = saved['params']['strf_alpha'][:,:,99]
    # # params['Strf_gain'] = saved['params']['strf_gain'][:,:,99]
    # # params['output_ad'] = saved['params']['output_ad'][:,:,99]
    # # params['on_ron_gSYN'] = saved['params']['on_ron_gsyn'][:,:,99]
    # # params['off_ron_gSYN'] = saved['params']['off_ron_gsyn'][:,:,99]
    # # params['on_sonoff_gSYN'] = saved['params']['on_sonoff_gsyn'][:,:,99]
    # # params['off_sonoff_gSYN'] = saved['params']['off_sonoff_gsyn'][:,:,99]
    # # params['sonoff_ron_gSYN'] = saved['params']['sonoff_ron_gsyn'][:,:,99]
    # # params['abs_ref'] = saved['params']['abs_ref'][:,:,99]
    # # params['rel_ref_a'] = saved['params']['rel_ref_a'][:,:,99]
    # # params['rel_ref_b'] = saved['params']['rel_ref_b'][:,:,99]
    # # params['rel_ref_c'] = saved['params']['rel_ref_c'][:,:,99]

    lrs['Strf_alpha'] = np.max(params['Strf_alpha'])*args['simulation']['lr']
    lrs['Strf_gain'] = np.max(params['Strf_gain'])*args['simulation']['lr']
    lrs['output_ad'] = np.max(params['output_ad'])*args['simulation']['lr']
    lrs['on_ron_gSYN'] = np.max(params['on_ron_gSYN'])*args['simulation']['lr']
    lrs['off_ron_gSYN'] = np.max(params['off_ron_gSYN'])*args['simulation']['lr']
    lrs['on_sonoff_gSYN'] = np.max(params['on_sonoff_gSYN'])*args['simulation']['lr']
    lrs['off_sonoff_gSYN'] = np.max(params['off_sonoff_gSYN'])*args['simulation']['lr']
    lrs['sonoff_ron_gSYN'] = np.max(params['sonoff_ron_gSYN'])*args['simulation']['lr']
    lrs['abs_ref'] = np.max(params['abs_ref'])*args['simulation']['lr']
    lrs['rel_ref_a'] = np.max(params['rel_ref_a'])*args['simulation']['lr']
    lrs['rel_ref_b'] = np.max(params['rel_ref_b'])*args['simulation']['lr']
    lrs['rel_ref_c'] = np.max(params['rel_ref_c'])*args['simulation']['lr']

    return {'params': params, 'lrs' : lrs}

