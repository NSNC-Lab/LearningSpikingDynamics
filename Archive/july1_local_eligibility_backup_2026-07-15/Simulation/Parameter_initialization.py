import numpy as np

def init_params(args):
    params = {}
    lrs = {}

    ###
    Strf_alpha_mag = 100
    Strf_gain_mag = 0.02
    output_ad_mag = 0.005
    on_ron_gSYN_mag = 0.05
    off_ron_gSYN_mag = 0.05
    on_sonoff_gSYN_mag = 0.05
    off_sonoff_gSYN_mag = 0.05
    sonoff_ron_gSYN_mag = 0.05
    abs_ref_mag = 2
    rel_ref_a_mag = 1
    rel_ref_b_mag = 10
    rel_ref_c_mag = 1

    ###
    Strf_alpha_min = 1
    Strf_gain_min = 0.001
    output_ad_min = 0.0001
    on_ron_gSYN_min = 0.01
    off_ron_gSYN_min = 0.01
    on_sonoff_gSYN_min = 0.01
    off_sonoff_gSYN_min = 0.01
    sonoff_ron_gSYN_min = 0.01
    abs_ref_min = 0.1
    rel_ref_a_min = 0.01
    rel_ref_b_min = 5
    rel_ref_c_min = 0.2

    params['Strf_alpha'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(Strf_alpha_mag-Strf_alpha_min)+Strf_alpha_min)
    params['Strf_gain'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(Strf_gain_mag-Strf_gain_min)+Strf_gain_min)
    params['output_ad'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(output_ad_mag-output_ad_min)+output_ad_min)
    params['on_ron_gSYN'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(on_ron_gSYN_mag-on_ron_gSYN_min)+on_ron_gSYN_min)
    params['off_ron_gSYN'] = (np.random.rand(args['simulation']['batch_size'],len(args['simulation']['cell_targets']))*(off_ron_gSYN_mag-off_ron_gSYN_min)+off_ron_gSYN_min)
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

