import copy
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from Simulation import Architecture_Declaration, Eligibility_handler, Loss_handler, Simulation_Handler, conditional_handler, ode_handler
from Pre_Processing import pre_cortical_handler


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_args():
    with (REPO_ROOT / "simulation_config.yaml").open("r", encoding="utf-8") as handle:
        args = copy.deepcopy(yaml.safe_load(handle))
    args["simulation"].update(
        {
            "batch_size": 1,
            "cell_targets": [1],
            "device": "cpu",
            "epochs": 1,
            "sim_len": 6,
            "PSTH_granularity": 3,
        }
    )
    args.setdefault("parameter_initialization", {}).setdefault("from_mat", {})["enabled"] = False
    return args


def test_params():
    scalar = np.ones((1, 1), dtype=np.float32)
    return {
        "Strf_alpha": scalar * 25.0,
        "Strf_gain": scalar * 0.01,
        "output_ad": scalar * 0.003,
        "on_ron_gSYN": scalar * 0.03,
        "off_ron_gSYN": scalar * 0.03,
        "on_sonoff_gSYN": scalar * 0.03,
        "off_sonoff_gSYN": scalar * 0.03,
        "sonoff_ron_gSYN": scalar * 0.03,
        "abs_ref": scalar * 5.0,
        "rel_ref_a": scalar * 1.0,
        "rel_ref_b": scalar * 1.0,
        "rel_ref_c": scalar * 0.5,
    }


class TrainingRegressionTests(unittest.TestCase):
    def setUp(self):
        self.args = test_args()
        self.states = Architecture_Declaration.build_network(
            self.args, torch.device("cpu"), test_params()
        )

    def test_invalid_end_to_end_refractory_training_fails_closed(self):
        self.args["optimization"]["train_refractory"] = True
        with self.assertRaisesRegex(
            NotImplementedError, "CV multiplier is not d\\(CV\\)/d\\(parameter\\)"
        ):
            Simulation_Handler.run_optimization(
                self.args,
                {"params": test_params(), "lrs": {}},
                {},
            )

    def test_negative_spike_sentinels_do_not_trigger_delayed_synapses(self):
        conditional_handler.condtion3(self.args, self.states, timestep=0)
        for dynamic in self.states["synapses"]["Dynamic"].values():
            self.assertTrue(torch.count_nonzero(dynamic["PSC_x"]) == 0)

    def test_zero_input_network_remains_at_rest_without_phantom_volley(self):
        steps = 100
        args = copy.deepcopy(self.args)
        args["simulation"]["sim_len"] = steps
        states = Architecture_Declaration.build_network(
            args, torch.device("cpu"), test_params()
        )
        zero_inputs = {
            "onset_offset_spks": {
                "onset_spks": torch.zeros((1, 10, 1, steps)),
                "offset_spks": torch.zeros((1, 10, 1, steps)),
            },
            "noise_spks": torch.zeros((10, 1, steps)),
        }
        for timestep in range(steps):
            ode_handler.run_odes(args, states, zero_inputs, timestep)
            conditional_handler.run_conditionals(args, states, timestep)

        for dynamic in states["neurons"]["Dynamic"].values():
            self.assertFalse(torch.any(dynamic["tspike"] >= 0))
        for dynamic in states["synapses"]["Dynamic"].values():
            self.assertTrue(torch.count_nonzero(dynamic["PSC_x"]) == 0)
            self.assertTrue(torch.count_nonzero(dynamic["PSC_s"]) == 0)

    def test_no_previous_spike_starts_fully_recovered(self):
        learnable = self.states["neurons"]["Learnable"]["ron"]
        with torch.no_grad():
            learnable["rel_ref_a"].fill_(0.001)
            learnable["rel_ref_b"].fill_(10.0)
            learnable["rel_ref_c"].fill_(0.2)
        relative_ref = conditional_handler.refractory_state(self.states, timestep=0)
        torch.testing.assert_close(
            relative_ref["probability"], torch.ones_like(relative_ref["probability"])
        )

    def test_synaptic_event_uses_current_recovered_facilitation_depression(self):
        synapse = self.states["synapses"]["Dynamic"]["on_ron"]
        onset = self.states["neurons"]["Dynamic"]["on"]
        synapse["PSC_F"][..., -1].fill_(0.8)
        synapse["PSC_P"][..., -1].fill_(0.5)
        synapse["PSC_q"][..., -1].fill_(0.1)
        delay = int(
            self.states["synapses"]["Static"]["on_ron"]["PSC_delay"]
            / self.args["simulation"]["dt"]
        )
        for timestep in range(delay + 1):
            onset["spike_mask"].fill_(timestep == 0)
            conditional_handler.condtion3(self.args, self.states, timestep=timestep)
        self.assertAlmostEqual(float(synapse["PSC_x"][0, 0, 0, -1]), 0.4, places=6)

    def test_lossless_delay_queues_preserve_bursts_and_soft_tangent_alignment(self):
        args = copy.deepcopy(self.args)
        burst_steps = 20
        longest_delay = 30
        args["simulation"]["sim_len"] = burst_steps + longest_delay + 1
        states = Architecture_Declaration.build_network(
            args, torch.device("cpu"), test_params()
        )
        first_layer_synapses = (
            "on_ron",
            "off_ron",
            "on_sonoff",
            "off_sonoff",
        )
        for synapse_name in first_layer_synapses:
            # Make every delivered event have unit amplitude so PSC_x is an
            # exact event counter in this no-ODE forward-only test.
            states["synapses"]["Static"][synapse_name]["PSC_fF"] = 0.0
            states["synapses"]["Static"][synapse_name]["PSC_fP"] = 0.0

        for timestep in range(args["simulation"]["sim_len"]):
            event = timestep < burst_steps
            for neuron_name in ("on", "off"):
                dynamic = states["neurons"]["Dynamic"][neuron_name]
                threshold = states["neurons"]["Static"][neuron_name]["V_thresh"]
                dynamic["V"][..., -1].fill_(threshold + 1.0 if event else threshold - 1.0)

            # Eligibility writes the soft event derivative before the hard
            # conditional pass.  Mirror that production ordering and verify
            # the soft tangent is read from the same delayed slot as the event.
            for synapse_name in first_layer_synapses:
                dynamic = states["synapses"]["Dynamic"][synapse_name]
                delay = int(
                    states["synapses"]["Static"][synapse_name]["PSC_delay"]
                    / args["simulation"]["dt"]
                )
                write_idx = timestep % (delay + 1)
                dynamic["dsoft_gain_delay"][..., write_idx].fill_(float(event))
                dynamic["dsoft_alpha_delay"][..., write_idx].zero_()

            conditional_handler.run_conditionals(args, states, timestep)

        # The refractory history intentionally retains only five timestamps;
        # all twenty pending events must nevertheless reach every 10- and
        # 30-step delayed first-layer synapse.
        for neuron_name in ("on", "off"):
            valid_history = states["neurons"]["Dynamic"][neuron_name]["tspike"] >= 0
            self.assertEqual(int(valid_history[0, 0, 0].sum()), 5)
        for synapse_name in first_layer_synapses:
            dynamic = states["synapses"]["Dynamic"][synapse_name]
            self.assertAlmostEqual(
                float(dynamic["PSC_x"][0, 0, 0, -1]), burst_steps, places=6
            )
            self.assertAlmostEqual(
                float(dynamic["dPSC_x_gain"][0, 0, 0, -1]),
                burst_steps,
                places=6,
            )

    def test_psth_bin_includes_exactly_current_completed_interval(self):
        spikes = self.states["neurons"]["Dynamic"]["ron"]["spikes_holder"]
        spikes[0, 0, 0, 0] = 1
        spikes[0, 0, 0, 2] = 1
        spikes[0, 0, 0, 3] = 1
        target = torch.tensor([[1.0, 0.0]])
        self.args["loss"]["psth_count_bias_correction"] = 0.0

        first = Loss_handler.calculate_loss(self.states, target, self.args, 2, 3)
        second = Loss_handler.calculate_loss(self.states, target, self.args, 5, 3)
        self.assertAlmostEqual(float(first["gradient"]), 2.0)
        self.assertAlmostEqual(float(second["gradient"]), 2.0)

    def test_count_bias_correction_shifts_multiplier_by_one(self):
        spikes = self.states["neurons"]["Dynamic"]["ron"]["spikes_holder"]
        spikes[0, 0, 0, 0] = 1
        target = torch.tensor([[1.0, 0.0]])
        self.args["loss"]["psth_count_bias_correction"] = 0.5
        result = Loss_handler.calculate_loss(self.states, target, self.args, 2, 3)
        self.assertAlmostEqual(float(result["gradient"]), -1.0)

    def test_relative_hazard_eligibilities_match_local_central_difference(self):
        timestep = 10
        dynamic = self.states["neurons"]["Dynamic"]["ron"]
        learnable = self.states["neurons"]["Learnable"]["ron"]
        static = self.states["neurons"]["Static"]["ron"]
        with torch.no_grad():
            dynamic["tspike"][..., 0] = 0
            dynamic["V"][..., -1] = static["V_thresh"]
            learnable["abs_ref"].fill_(0.0)
            learnable["rel_ref_a"].fill_(0.1)
            learnable["rel_ref_b"].fill_(2.0)
            learnable["rel_ref_c"].fill_(0.4)

        zero_rate = torch.zeros((timestep + 1, 1, 1))
        rates = {
            "onset_rate": zero_rate,
            "offset_rate": zero_rate,
            "onset_rate_gain_deriv": zero_rate,
            "offset_rate_gain_deriv": zero_rate,
            "onset_rate_deriv": zero_rate,
            "offset_rate_deriv": zero_rate,
        }
        Eligibility_handler.update_eligibility(self.args, self.states, rates, timestep)

        width = self.args["eligibility"]["surrogate_width_mv"]
        voltage_gate = 0.5 * (1.0 + np.tanh(0.0 / width))
        recovered = 0.5 * (1.0 + np.tanh(float(timestep)))

        def soft_expected_spike(a, b, c):
            raw = c * (np.tanh(a * timestep - b) + 1.0)
            return voltage_gate * recovered * np.clip(raw, 0.0, 1.0)

        base = [0.1, 2.0, 0.4]
        epsilons = [1e-5, 1e-5, 1e-5]
        names = ["rel_ref_a", "rel_ref_b", "rel_ref_c"]
        for index, (name, epsilon) in enumerate(zip(names, epsilons)):
            plus = base.copy()
            minus = base.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            finite_difference = (
                soft_expected_spike(*plus) - soft_expected_spike(*minus)
            ) / (2.0 * epsilon)
            analytic = float(learnable[f"{name}_accum"][0, 0]) / 10.0
            self.assertAlmostEqual(analytic, finite_difference, places=4)

    def test_causal_offset_has_no_future_maximum_plateau_and_matches_fd(self):
        base = np.array([0.0, 1.0, 2.0, 1.5, 0.5, 3.0, 2.0], dtype=np.float64)[:, None, None]
        direction = np.array([0.2, -0.1, 0.3, 0.4, -0.2, 0.1, -0.3], dtype=np.float64)[:, None, None]
        zeros = np.zeros_like(base)
        offset, derivative, _ = pre_cortical_handler.causal_running_max_offset(
            base, direction, zeros
        )
        np.testing.assert_allclose(offset[:3], 0.0)

        epsilon = 1e-6
        plus = pre_cortical_handler.causal_running_max_offset(
            base + epsilon * direction, zeros, zeros
        )[0]
        minus = pre_cortical_handler.causal_running_max_offset(
            base - epsilon * direction, zeros, zeros
        )[0]
        finite_difference = (plus - minus) / (2.0 * epsilon)
        np.testing.assert_allclose(derivative, finite_difference, rtol=1e-8, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
