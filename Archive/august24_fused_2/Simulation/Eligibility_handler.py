import torch

from Simulation.fused_helpers import absolute_release, relative_terms


SYNAPSE_INDEX = {"on_ron":3, "off_ron":4, "on_sonoff":5, "off_sonoff":6, "sonoff_ron":7}


def update_eligibility(args, states, rate_object, timestep):
    dt = args['simulation']['dt']
    width = args['simulation'].get('surrogate_width_mv', 5.0)

    for name, static in states['neurons']['Static'].items():
        voltage = states['neurons']['Dynamic'][name]['V'][:,:,:,-1]
        x = (voltage-static['V_thresh'])/width
        static['q'], static['psi'] = 0.5*(1+torch.tanh(x)), (1-torch.tanh(x).square())/(2*width)

    ron_s, ron_d, ron_l = states['neurons']['Static']['ron'], states['neurons']['Dynamic']['ron'], states['neurons']['Learnable']['ron']
    last_spike = ron_d['tspike'].max(dim=-1).values
    delta = timestep-last_spike
    probability, dp_da, dp_db, dp_dc = relative_terms(delta, ron_l['rel_ref_a'][:,None,:], ron_l['rel_ref_b'][:,None,:], ron_l['rel_ref_c'][:,None,:])
    release, drelease = absolute_release(delta, ron_l['abs_ref'][:,None,:], dt)
    states['neurons']['Learnable']['Bk'] = -states['synapses']['Learnable']['sonoff_ron']['gSYN']*(ron_s['V_thresh']-states['synapses']['Static']['sonoff_ron']['ESYN'])

    event = torch.zeros((*ron_s['q'].shape,args['simulation']['num_params']), device=ron_s['q'].device)
    for name, static in states['synapses']['Static'].items():
        post = name.split('_')[1]
        post_s, post_d = states['neurons']['Static'][post], states['neurons']['Dynamic'][post]
        local = dt*post_s['psi']*-post_s['R']*states['synapses']['Dynamic'][name]['PSC_s'][:,:,:,-2]*(post_d['V'][:,:,:,-2]-static['ESYN'])/post_s['tau']
        if post != 'ron':
            local = local*(~torch.any(timestep <= post_d['tspike']+post_s['t_ref']/dt,dim=-1))
        local = local*release*probability
        states['synapses']['Learnable'][name]['gSYN_accum'] += local.sum(dim=1)
        event[...,SYNAPSE_INDEX[name]] = local if post == 'ron' else local*states['neurons']['Learnable']['Bk'][:,None,:]

    onset_s, onset_d = states['neurons']['Static']['on'], states['neurons']['Dynamic']['on']
    offset_s, offset_d = states['neurons']['Static']['off'], states['neurons']['Dynamic']['off']
    scale = (dt/1000)*dt
    gain_event = scale*((onset_s['psi']*-onset_s['R']*onset_s['g_postIC']*rate_object['onset_rate_gain_deriv'][timestep,:,None,:]*(onset_d['V'][:,:,:,-2]-onset_s['E_exc'])/onset_s['tau']) +
                        (offset_s['psi']*-offset_s['R']*offset_s['g_postIC']*rate_object['offset_rate_gain_deriv'][timestep,:,None,:]*(offset_d['V'][:,:,:,-2]-offset_s['E_exc'])/offset_s['tau']))
    alpha_event = scale*((onset_s['psi']*-onset_s['R']*onset_s['g_postIC']*rate_object['onset_rate_deriv'][timestep,:,None,:]*(onset_d['V'][:,:,:,-2]-onset_s['E_exc'])/onset_s['tau']) +
                         (offset_s['psi']*-offset_s['R']*offset_s['g_postIC']*rate_object['offset_rate_deriv'][timestep,:,None,:]*(offset_d['V'][:,:,:,-2]-offset_s['E_exc'])/offset_s['tau']))
    gain_event, alpha_event = gain_event*release*probability, alpha_event*release*probability
    states['neurons']['Learnable']['STRF_gain_accum'] += gain_event.sum(dim=1)
    states['neurons']['Learnable']['STRF_alpha_accum'] += alpha_event.sum(dim=1)
    event[...,0], event[...,1] = gain_event, alpha_event

    trace = ron_l['output_ad_trace']
    output_event = release*probability*ron_s['psi']*dt*(-ron_s['R']*(ron_d['V'][:,:,:,-2]-ron_s['E_k'])/ron_s['tau'])*trace
    ron_l['output_ad_accum'] += output_event.sum(dim=1)
    trace.mul_(1-dt/ron_s['tau_ad'])
    event[...,2] = output_event

    ref_events = (ron_s['q']*probability*drelease,
                  release*ron_s['q']*dp_da,
                  release*ron_s['q']*dp_db,
                  release*ron_s['q']*dp_dc)
    for index, (name, value) in enumerate(zip(('abs_ref','rel_ref_a','rel_ref_b','rel_ref_c'),ref_events), start=8):
        ron_l[f'{name}_accum'] += value.sum(dim=1)
        event[...,index] = value

    cv = states['neurons']['CV']
    cv['q_window'][...,:-1] = cv['q_window'][...,1:].clone()
    cv['dq_window'][...,:-1,:] = cv['dq_window'][...,1:,:].clone()
    cv['q_window'][...,-1] = release*probability*ron_s['q']
    cv['dq_window'][...,-1,:] = event
    return states


def update_cv_events(args, states, timestep):
    if timestep < 5:
        return states
    spikes = states['neurons']['Dynamic']['ron']['spikes_holder'][...,timestep-5].bool()
    if not spikes.any().item():
        return states

    cv = states['neurons']['CV']
    batch, trial, cell = torch.where(spikes)
    q = cv['q_window'][batch,trial,cell]
    dq = cv['dq_window'][batch,trial,cell]
    denominator = q.sum(dim=1)
    times = torch.arange(timestep-10,timestep+1,device=q.device,dtype=q.dtype)*args['simulation']['dt']
    soft_time = (q*times).sum(dim=1)/denominator.clamp_min(1e-8)
    ds = ((times[None,:,None]-soft_time[:,None,None])*dq).sum(dim=1)/denominator.clamp_min(1e-8)[:,None]
    valid = (denominator > 1e-8) & torch.isfinite(ds).all(dim=1)
    batch, trial, cell, ds = batch[valid], trial[valid], cell[valid], ds[valid]
    if batch.numel() == 0:
        return states

    hard_time = torch.full((batch.numel(),),(timestep-5)*args['simulation']['dt'],device=q.device,dtype=q.dtype)
    previous_time = cv['previous_time'][batch,trial,cell]
    previous_ds = cv['previous_ds'][batch,trial,cell]
    has_previous = previous_time >= 0
    if has_previous.any().item():
        b, c = batch[has_previous], cell[has_previous]
        isi = hard_time[has_previous]-previous_time[has_previous]
        disi = ds[has_previous]-previous_ds[has_previous]
        cv['n'].index_put_((b,c),torch.ones_like(isi),accumulate=True)
        cv['sum'].index_put_((b,c),isi,accumulate=True)
        cv['sum_sq'].index_put_((b,c),isi.square(),accumulate=True)
        cv['dsum'].index_put_((b,c),disi,accumulate=True)
        cv['dsum_sq'].index_put_((b,c),2*isi[:,None]*disi,accumulate=True)
    cv['previous_time'][batch,trial,cell] = hard_time
    cv['previous_ds'][batch,trial,cell] = ds
    return states
