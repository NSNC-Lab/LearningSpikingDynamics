import torch

from Simulation.fused_helpers import cv_from_stats, rate_terms


def handle_loss(args, states, gt_data, timestep, epoch):
    granularity = args['simulation']['PSTH_granularity']
    if (timestep+1) % granularity == 0:
        result = calculate_loss(states,gt_data['psth_holder'],args,timestep,epoch)
        update_grad(states,result['gradient'])
    if timestep == args['simulation']['sim_len']-1:
        cv = calculate_CV_loss(states,gt_data['raster_holder'],args,epoch)
        update_grad_CV_and_rate(states,cv['gradient'],gt_data['psth_holder'],args,granularity,epoch)
    return states


def calculate_loss(states, target, args, timestep, epoch):
    end = timestep+1
    start = end-args['simulation']['PSTH_granularity']
    target_index = end//args['simulation']['PSTH_granularity']-1
    simulated = states['neurons']['Dynamic']['ron']['spikes_holder'][...,start:end].sum((1,3))
    residual = simulated-target[None,:,target_index]
    sse, psth_objective = residual.square(), (residual-0.5).square()
    states['neurons']['BookKeeping']['sse_loss'][:,:,epoch] += sse
    states['neurons']['BookKeeping']['psth_loss'][:,:,epoch] += psth_objective
    states['neurons']['Dynamic']['ron']['mean_sse_loss'] += sse.mean().cpu()
    return {'loss':sse,'gradient':2*(residual-0.5)}


def calculate_CV_loss(states, target, args, epoch):
    cv_state = states['neurons']['CV']
    simulated_cv, dcv, valid = cv_from_stats(cv_state['n'],cv_state['sum'],cv_state['sum_sq'],cv_state['dsum'],cv_state['dsum_sq'])
    target_cv = torch.zeros(target.shape[0],device=target.device)
    target_valid = torch.zeros(target.shape[0],device=target.device,dtype=torch.bool)
    analysis_end = args['simulation']['sim_len']-5
    for cell in range(target.shape[0]):
        intervals = [torch.diff(torch.where(target[cell,trial,:analysis_end] == 1)[0]).float()*args['simulation']['dt'] for trial in range(target.shape[1])]
        intervals = [value for value in intervals if value.numel()]
        if intervals:
            intervals = torch.cat(intervals)
            if intervals.numel() > 1 and intervals.mean() > 0:
                target_cv[cell] = intervals.std()/intervals.mean()
                target_valid[cell] = torch.isfinite(target_cv[cell])
    residual = simulated_cv-target_cv[None]
    loss = args['simulation']['lamda2']*residual.square()*(valid & target_valid[None])
    gradient = 2*args['simulation']['lamda2']*residual[...,None]*dcv
    gradient *= (valid & target_valid[None])[...,None]
    states['neurons']['BookKeeping']['cv_loss'][:,:,epoch] = loss
    states['neurons']['Dynamic']['ron']['mean_CV_loss'] = loss.mean().cpu()
    return {'loss':loss,'gradient':gradient}


def update_grad(states, gradient):
    for name in states['synapses']['Static']:
        learnable = states['synapses']['Learnable'][name]
        eligibility = learnable['gSYN_accum']
        if name.rsplit('_',1)[1] != 'ron':
            eligibility = eligibility*states['neurons']['Learnable']['Bk']
        learnable['gSYN_grad'] += gradient*eligibility
        learnable['gSYN_accum_rate'] += eligibility
        learnable['gSYN_accum'].zero_()
    ron = states['neurons']['Learnable']['ron']
    for owner, name in ((ron,'output_ad'),(states['neurons']['Learnable'],'STRF_alpha'),(states['neurons']['Learnable'],'STRF_gain'),
                        (ron,'abs_ref'),(ron,'rel_ref_a'),(ron,'rel_ref_b'),(ron,'rel_ref_c')):
        owner[f'{name}_grad'] += gradient*owner[f'{name}_accum']
        owner[f'{name}_accum_rate'] += owner[f'{name}_accum']
        owner[f'{name}_accum'].zero_()
    return states


def parameter_grad_views(states):
    neurons, ron, synapses = states['neurons']['Learnable'], states['neurons']['Learnable']['ron'], states['synapses']['Learnable']
    return [(neurons['STRF_gain_grad'],neurons['STRF_gain_accum_rate']),
            (neurons['STRF_alpha_grad'],neurons['STRF_alpha_accum_rate']),
            (ron['output_ad_grad'],ron['output_ad_accum_rate']),
            *[(synapses[name]['gSYN_grad'],synapses[name]['gSYN_accum_rate']) for name in ('on_ron','off_ron','on_sonoff','off_sonoff','sonoff_ron')],
            (ron['abs_ref_grad'],ron['abs_ref_accum_rate']),
            (ron['rel_ref_a_grad'],ron['rel_ref_a_accum_rate']),
            (ron['rel_ref_b_grad'],ron['rel_ref_b_accum_rate']),
            (ron['rel_ref_c_grad'],ron['rel_ref_c_accum_rate'])]


def update_grad_CV_and_rate(states, cv_gradient, target, args, granularity, epoch):
    full_len = target.shape[-1]*granularity
    exposure = states['neurons']['Dynamic']['ron']['spikes_holder'].shape[1]*full_len*args['simulation']['dt']/1000
    simulated_count = states['neurons']['Dynamic']['ron']['spikes_holder'][...,:full_len].sum((1,3)).float()
    target_count = target.sum(-1)[None]
    rate_loss, rate_gradient = rate_terms(simulated_count,target_count,exposure,args['simulation']['lamda1'])
    states['neurons']['BookKeeping']['rate_loss'][:,:,epoch] = rate_loss
    states['neurons']['BookKeeping']['fused_loss'][:,:,epoch] = states['neurons']['BookKeeping']['psth_loss'][:,:,epoch]+states['neurons']['BookKeeping']['cv_loss'][:,:,epoch]+rate_loss
    for index, (gradient, count_eligibility) in enumerate(parameter_grad_views(states)):
        gradient += rate_gradient*count_eligibility+cv_gradient[...,index]
        count_eligibility.zero_()
    return states
