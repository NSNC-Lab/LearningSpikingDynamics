import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from Simulation import Architecture_Declaration, Eligibility_handler, Loss_handler, conditional_handler, ode_handler, update_handler
from Simulation.fused_helpers import absolute_release, cv_from_stats, rate_terms, relative_terms


class FusedFixTests(unittest.TestCase):
    def test_rate_objective_matches_count_gradient(self):
        count = torch.tensor([[30.0]],dtype=torch.float64)
        target = torch.tensor([[12.0]],dtype=torch.float64)
        exposure, weight, eps = 4.0, 3.0, 1e-5
        loss, gradient = rate_terms(count,target,exposure,weight)
        plus = rate_terms(count+eps,target,exposure,weight)[0]
        minus = rate_terms(count-eps,target,exposure,weight)[0]
        self.assertTrue(torch.allclose(gradient,(plus-minus)/(2*eps),rtol=1e-7,atol=1e-9))
        self.assertTrue(torch.allclose(loss,weight*(count/exposure-target/exposure).square()))

    def test_relative_probability_is_bounded_and_derivatives_match_fd(self):
        delta = torch.tensor([[[2.0]]],dtype=torch.float64)
        a, b, c = (torch.tensor([[[value]]],dtype=torch.float64) for value in (0.3,0.4,0.8))
        probability, da, db, dc = relative_terms(delta,a,b,c)
        eps = 1e-6
        for analytic, values, index in ((da,[a,b,c],0),(db,[a,b,c],1),(dc,[a,b,c],2)):
            plus, minus = values.copy(), values.copy()
            plus[index], minus[index] = values[index]+eps, values[index]-eps
            fd = (relative_terms(delta,*plus)[0]-relative_terms(delta,*minus)[0])/(2*eps)
            self.assertTrue(torch.allclose(analytic,fd,rtol=1e-5,atol=1e-7))
        self.assertTrue(torch.allclose(probability,0.5*c*(1+torch.tanh(a*delta-b))))
        saturated = relative_terms(delta,torch.tensor([[[20.0]]]),torch.zeros(1,1,1),2*torch.ones(1,1,1))
        self.assertTrue((probability >= 0).all() and (probability <= 1).all())
        self.assertEqual(sum(value.abs().sum().item() for value in saturated[1:]),0)

    def test_absolute_refractory_uses_physical_units(self):
        delta, abs_ref, dt = torch.tensor([[[15.0]]],dtype=torch.float64), torch.tensor([[[1.5]]],dtype=torch.float64), 0.1
        release, derivative = absolute_release(delta,abs_ref,dt)
        eps = 1e-6
        fd = (absolute_release(delta,abs_ref+eps,dt)[0]-absolute_release(delta,abs_ref-eps,dt)[0])/(2*eps)
        self.assertAlmostEqual(release.item(),0.5,places=12)
        self.assertTrue(torch.allclose(derivative,fd,rtol=1e-5,atol=1e-7))

    def test_cv_sufficient_statistics_match_fd(self):
        intervals = torch.tensor([1.2,2.5,4.1],dtype=torch.float64)
        direction = torch.tensor([0.3,-0.2,0.5],dtype=torch.float64)
        n = torch.tensor([[3.0]],dtype=torch.float64)
        total, total_sq = intervals.sum().reshape(1,1), intervals.square().sum().reshape(1,1)
        dtotal = direction.sum().reshape(1,1,1)
        dtotal_sq = (2*intervals*direction).sum().reshape(1,1,1)
        cv, dcv, valid = cv_from_stats(n,total,total_sq,dtotal,dtotal_sq)
        eps = 1e-6
        def sample_cv(values):
            return values.std(correction=1)/values.mean()
        fd = (sample_cv(intervals+eps*direction)-sample_cv(intervals-eps*direction))/(2*eps)
        self.assertTrue(valid.item())
        self.assertTrue(torch.allclose(cv,sample_cv(intervals).reshape(1,1)))
        self.assertTrue(torch.allclose(dcv.squeeze(),fd,rtol=1e-5,atol=1e-7))

    def test_strf_reduction_preserves_batch_and_cell(self):
        args = {'simulation':{'batch_size':2,'cell_targets':[1,2],'epochs':1,'sim_len':21,'num_params':12,'dt':0.1,'device':'cpu'}}
        shape = (2,2)
        params = {name:np.ones(shape,np.float32) for name in ('Strf_gain','Strf_alpha','output_ad','abs_ref','rel_ref_a','rel_ref_b','rel_ref_c','on_ron_gSYN','off_ron_gSYN','on_sonoff_gSYN','off_sonoff_gSYN','sonoff_ron_gSYN')}
        states = Architecture_Declaration.build_network(args,torch.device('cpu'),params)
        for name in ('on','off'):
            states['neurons']['Dynamic'][name]['V'][...,-1] = states['neurons']['Static'][name]['V_thresh']
        states['neurons']['Dynamic']['ron']['V'][...,-1] = -60
        zeros = torch.zeros((1,2,2))
        one = zeros.clone(); one[0,0,0] = 1
        rates = {'onset_rate_gain_deriv':one,'offset_rate_gain_deriv':zeros,'onset_rate_deriv':zeros,'offset_rate_deriv':zeros}
        Eligibility_handler.update_eligibility(args,states,rates,0)
        gain = states['neurons']['Learnable']['STRF_gain_accum']
        self.assertEqual(tuple(gain.shape),(2,2))
        self.assertNotEqual(gain[0,0].item(),0)
        self.assertEqual(torch.count_nonzero(gain).item(),1)

    def test_psi_is_cached_before_spike_reset(self):
        args = {'simulation':{'batch_size':1,'cell_targets':[1],'epochs':1,'sim_len':21,'num_params':12,'dt':0.1,'device':'cpu','surrogate_width_mv':5.0}}
        shape = (1,1)
        params = {name:np.ones(shape,np.float32) for name in ('Strf_gain','Strf_alpha','output_ad','abs_ref','rel_ref_a','rel_ref_b','rel_ref_c','on_ron_gSYN','off_ron_gSYN','on_sonoff_gSYN','off_sonoff_gSYN','sonoff_ron_gSYN')}
        states = Architecture_Declaration.build_network(args,torch.device('cpu'),params)
        states['neurons']['Dynamic']['ron']['V'][...,-1] = states['neurons']['Static']['ron']['V_thresh']
        zeros = torch.zeros((1,1,1))
        rates = {name:zeros for name in ('onset_rate_gain_deriv','offset_rate_gain_deriv','onset_rate_deriv','offset_rate_deriv')}
        Eligibility_handler.update_eligibility(args,states,rates,0)
        conditional_handler.condtion2(args,states,0)
        self.assertAlmostEqual(states['neurons']['Static']['ron']['psi'].mean().item(),0.1,places=6)
        self.assertTrue(torch.all(states['neurons']['Dynamic']['ron']['V'][...,-1] == states['neurons']['Static']['ron']['V_reset']))

    def test_eligibility_uses_old_euler_state_and_honors_hidden_clamp(self):
        args = {'simulation':{'batch_size':1,'cell_targets':[1],'epochs':1,'sim_len':31,'num_params':12,'dt':0.1,'device':'cpu'}}
        params = {name:np.ones((1,1),np.float32) for name in ('Strf_gain','Strf_alpha','output_ad','abs_ref','rel_ref_a','rel_ref_b','rel_ref_c','on_ron_gSYN','off_ron_gSYN','on_sonoff_gSYN','off_sonoff_gSYN','sonoff_ron_gSYN')}
        states = Architecture_Declaration.build_network(args,torch.device('cpu'),params)
        ron, sonoff = states['neurons']['Dynamic']['ron'], states['neurons']['Dynamic']['sonoff']
        ron['V'][...,-2], ron['V'][...,-1] = -60, states['neurons']['Static']['ron']['V_thresh']
        states['synapses']['Dynamic']['on_ron']['PSC_s'][...,-2], states['synapses']['Dynamic']['on_ron']['PSC_s'][...,-1] = 1,100
        sonoff['V'][...,-1], sonoff['tspike'][...,-1] = states['neurons']['Static']['sonoff']['V_thresh'],20
        states['synapses']['Dynamic']['on_sonoff']['PSC_s'][...,-2] = 1
        zeros = torch.zeros((31,1,1)); rates = {name:zeros for name in ('onset_rate_gain_deriv','offset_rate_gain_deriv','onset_rate_deriv','offset_rate_deriv')}
        Eligibility_handler.update_eligibility(args,states,rates,20)
        expected = 10*.1*.1*-200*(-60)/20
        self.assertAlmostEqual(states['synapses']['Learnable']['on_ron']['gSYN_accum'].item(),expected,places=4)
        self.assertEqual(states['synapses']['Learnable']['on_sonoff']['gSYN_accum'].item(),0)

    def test_tiny_fused_training_path_is_finite(self):
        args = {'simulation':{'batch_size':1,'cell_targets':[1],'epochs':1,'sim_len':31,'num_params':12,'dt':0.1,'device':'cpu','PSTH_granularity':10,'lamda1':10.0,'lamda2':100.0,'surrogate_width_mv':5.0},'Adam':{'beta1':0.01,'beta2':0.9995}}
        values = {'Strf_gain':.015,'Strf_alpha':20,'output_ad':.001,'abs_ref':.1,'rel_ref_a':1,'rel_ref_b':0,'rel_ref_c':1,
                  'on_ron_gSYN':.01,'off_ron_gSYN':.01,'on_sonoff_gSYN':.01,'off_sonoff_gSYN':.01,'sonoff_ron_gSYN':.01}
        params = {name:np.full((1,1),value,np.float32) for name,value in values.items()}
        states = Architecture_Declaration.build_network(args,torch.device('cpu'),params)
        spike_object = {'onset_offset_spks':{'onset_spks':torch.zeros(1,10,1,31),'offset_spks':torch.zeros(1,10,1,31)},'noise_spks':torch.zeros(10,1,31)}
        zeros = torch.zeros(31,1,1)
        rates = {name:zeros for name in ('onset_rate_gain_deriv','offset_rate_gain_deriv','onset_rate_deriv','offset_rate_deriv')}
        target_raster = torch.zeros(1,10,31)
        target_raster[:,:,torch.tensor([2,9,17,26])] = 1
        target_psth = target_raster[...,:30].reshape(1,10,3,10).sum((1,3))
        target = {'raster_holder':target_raster,'psth_holder':target_psth}
        for timestep in range(31):
            ode_handler.run_odes(args,states,spike_object,timestep)
            conditional_handler.condtion1(args,states,timestep)
            if timestep in (2,8,15):
                states['neurons']['Dynamic']['ron']['V'][...,-1] = states['neurons']['Static']['ron']['V_thresh']+1
            Eligibility_handler.update_eligibility(args,states,rates,timestep)
            conditional_handler.condtion2(args,states,timestep)
            Eligibility_handler.update_cv_events(args,states,timestep)
            conditional_handler.condtion3(args,states,timestep)
            Loss_handler.handle_loss(args,states,target,timestep,0)
        lrs = {name:torch.tensor(1e-4) for name in values}
        update_handler.run_adam(states,args,lrs)
        self.assertEqual(states['neurons']['CV']['n'].item(),20)
        self.assertGreater(states['neurons']['BookKeeping']['cv_loss'].item(),0)
        self.assertGreater(states['neurons']['BookKeeping']['rate_loss'].item(),0)
        self.assertTrue(torch.allclose(states['neurons']['BookKeeping']['fused_loss'],states['neurons']['BookKeeping']['psth_loss']+states['neurons']['BookKeeping']['cv_loss']+states['neurons']['BookKeeping']['rate_loss']))
        self.assertTrue(torch.isfinite(states['neurons']['BookKeeping']['fused_loss']).all())
        self.assertGreater(states['neurons']['Adam']['m'].abs().sum().item(),0)
        self.assertTrue(torch.isfinite(states['neurons']['Adam']['m']).all() and torch.isfinite(states['neurons']['Adam']['v']).all())


if __name__ == '__main__':
    unittest.main()
