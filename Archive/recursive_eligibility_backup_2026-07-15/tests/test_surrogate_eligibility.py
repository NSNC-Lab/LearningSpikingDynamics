import copy
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from Simulation import (
    Architecture_Declaration,
    Eligibility_handler,
    conditional_handler,
    ode_handler,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_args(sim_len=16):
    with (REPO_ROOT / "simulation_config.yaml").open("r", encoding="utf-8") as handle:
        args = copy.deepcopy(yaml.safe_load(handle))
    args["simulation"].update(
        {
            "batch_size": 1,
            "cell_targets": [1],
            "device": "cpu",
            "epochs": 1,
            "sim_len": sim_len,
            "PSTH_granularity": min(10, sim_len),
        }
    )
    args.setdefault("parameter_initialization", {}).setdefault("from_mat", {})[
        "enabled"
    ] = False
    return args


def _make_params():
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
        "abs_ref": scalar * 0.0,
        "rel_ref_a": scalar * 0.1,
        "rel_ref_b": scalar * 2.0,
        "rel_ref_c": scalar * 0.4,
    }


def _build(args):
    return Architecture_Declaration.build_network(
        args, torch.device("cpu"), _make_params()
    )


def _zero_rates(steps):
    zeros = torch.zeros((steps, 1, 1), dtype=torch.float32)
    return {
        "onset_rate": zeros.clone(),
        "offset_rate": zeros.clone(),
        "onset_rate_gain_deriv": zeros.clone(),
        "offset_rate_gain_deriv": zeros.clone(),
        "onset_rate_deriv": zeros.clone(),
        "offset_rate_deriv": zeros.clone(),
    }


def _smooth_threshold(voltage, threshold, width):
    return 0.5 * (1.0 + np.tanh((voltage - threshold) / width))


class SurrogateEligibilityTests(unittest.TestCase):
    def test_normalized_conditional_mean_matches_scalar_finite_difference(self):
        threshold = -47.0
        width = 5.0
        voltage = -46.0
        voltage_direction = 1.7
        epsilon = 1e-5

        sech_squared = 1.0 - np.tanh((voltage - threshold) / width) ** 2
        for relative_probability in (0.0, 0.2, 1.0):
            plus = relative_probability * _smooth_threshold(
                voltage + epsilon * voltage_direction, threshold, width
            )
            minus = relative_probability * _smooth_threshold(
                voltage - epsilon * voltage_direction, threshold, width
            )
            finite_difference = (plus - minus) / (2.0 * epsilon)
            analytic = (
                relative_probability
                * 0.5
                * sech_squared
                / width
                * voltage_direction
            )
            self.assertAlmostEqual(analytic, finite_difference, places=10)

    def _output_accumulation(self, history, a, b, c, timestep=10):
        args = _make_args()
        states = _build(args)
        dynamic = states["neurons"]["Dynamic"]["ron"]
        static = states["neurons"]["Static"]["ron"]
        learnable = states["neurons"]["Learnable"]["ron"]
        direction = 1.75

        with torch.no_grad():
            dynamic["V"][..., -1].fill_(static["V_thresh"])
            if history:
                dynamic["tspike"][..., 0] = 0.0
            learnable["abs_ref"].zero_()
            learnable["rel_ref_a"].fill_(a)
            learnable["rel_ref_b"].fill_(b)
            learnable["rel_ref_c"].fill_(c)
            states["synapses"]["Dynamic"]["on_ron"]["dV_gSYN"][
                ..., -1
            ].fill_(direction)

        relative_probability = conditional_handler.refractory_state(
            states, timestep
        )["probability"].clone()
        Eligibility_handler.update_eligibility(
            args, states, _zero_rates(args["simulation"]["sim_len"]), timestep
        )
        actual = states["synapses"]["Learnable"]["on_ron"]["gSYN_accum"]
        width = float(args["eligibility"]["surrogate_width_mv"])
        expected = 10.0 * float(relative_probability[0, 0, 0]) * 0.5 / width * direction
        return float(actual[0, 0]), expected, float(relative_probability[0, 0, 0])

    def test_output_eligibility_uses_interior_relative_probability(self):
        actual, expected, probability = self._output_accumulation(
            history=True, a=0.1, b=2.0, c=0.4
        )
        self.assertGreater(probability, 0.0)
        self.assertLess(probability, 1.0)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_no_history_forces_output_probability_to_one(self):
        actual, expected, probability = self._output_accumulation(
            history=False, a=0.0, b=0.0, c=0.0
        )
        self.assertEqual(probability, 1.0)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_clamped_output_probabilities_zero_and_one(self):
        zero_actual, zero_expected, zero_probability = self._output_accumulation(
            history=True, a=0.0, b=0.0, c=0.0
        )
        one_actual, one_expected, one_probability = self._output_accumulation(
            history=True, a=1.0, b=0.0, c=1.0
        )
        self.assertEqual(zero_probability, 0.0)
        self.assertEqual(one_probability, 1.0)
        self.assertAlmostEqual(zero_actual, zero_expected, places=7)
        self.assertAlmostEqual(one_actual, one_expected, places=6)

    def test_input_spike_surrogate_does_not_use_output_relative_probability(self):
        timestep = 10
        args = _make_args()
        states = _build(args)
        onset_dynamic = states["neurons"]["Dynamic"]["on"]
        onset_static = states["neurons"]["Static"]["on"]
        ron_dynamic = states["neurons"]["Dynamic"]["ron"]
        ron_learnable = states["neurons"]["Learnable"]["ron"]
        direction = 2.0

        with torch.no_grad():
            onset_dynamic["V"][..., -1].fill_(onset_static["V_thresh"])
            onset_dynamic["dV_STRF_gain"][..., -1].fill_(direction)
            # Preserve the seeded direction during the eligibility-side Euler
            # update, isolating the threshold surrogate in this test.
            onset_dynamic["input_V_jacobian"].zero_()

            # The output gate is exactly zero, but it must not attenuate an
            # input neuron's deterministic threshold surrogate.
            ron_dynamic["tspike"][..., 0] = 0.0
            ron_learnable["rel_ref_a"].zero_()
            ron_learnable["rel_ref_b"].zero_()
            ron_learnable["rel_ref_c"].zero_()

        self.assertEqual(
            float(conditional_handler.refractory_state(states, timestep)["probability"][0, 0, 0]),
            0.0,
        )
        Eligibility_handler.update_eligibility(
            args, states, _zero_rates(args["simulation"]["sim_len"]), timestep
        )

        synapse = states["synapses"]["Dynamic"]["on_ron"]
        delay_steps = int(
            states["synapses"]["Static"]["on_ron"]["PSC_delay"]
            / args["simulation"]["dt"]
        )
        write_index = timestep % (delay_steps + 1)
        expected = 0.5 / float(args["eligibility"]["surrogate_width_mv"]) * direction
        torch.testing.assert_close(
            synapse["dsoft_gain_delay"][..., write_index],
            torch.full_like(synapse["dsoft_gain_delay"][..., write_index], expected),
        )

    def test_eligibility_is_measured_post_ode_and_before_hard_reset(self):
        args = _make_args(sim_len=2)
        states = _build(args)
        ron_dynamic = states["neurons"]["Dynamic"]["ron"]
        ron_static = states["neurons"]["Static"]["ron"]
        direction = 1.25

        with torch.no_grad():
            ron_dynamic["V"][..., -1].fill_(ron_static["V_thresh"] + 5.0)
            states["synapses"]["Dynamic"]["on_ron"]["dV_gSYN"][
                ..., -1
            ].fill_(direction)

        zero_inputs = {
            "onset_offset_spks": {
                "onset_spks": torch.zeros((1, 10, 1, 2)),
                "offset_spks": torch.zeros((1, 10, 1, 2)),
            },
            "noise_spks": torch.zeros((10, 1, 2)),
        }
        ode_handler.run_odes(args, states, zero_inputs, timestep=0)

        voltage_before_reset = ron_dynamic["V"][..., -1].clone()
        sensitivity_before_reset = states["synapses"]["Dynamic"]["on_ron"][
            "dV_gSYN"
        ][..., -1].clone()
        self.assertTrue(torch.all(voltage_before_reset > ron_static["V_thresh"]))

        Eligibility_handler.update_eligibility(
            args, states, _zero_rates(args["simulation"]["sim_len"]), timestep=0
        )
        width = float(args["eligibility"]["surrogate_width_mv"])
        offset = (voltage_before_reset - ron_static["V_thresh"]) / width
        expected = torch.sum(
            0.5 * (1.0 - torch.tanh(offset) ** 2) / width
            * sensitivity_before_reset,
            dim=1,
        )
        torch.testing.assert_close(
            states["synapses"]["Learnable"]["on_ron"]["gSYN_accum"], expected
        )

        conditional_handler.run_conditionals(args, states, timestep=0)
        torch.testing.assert_close(
            ron_dynamic["V"][..., -1],
            torch.full_like(ron_dynamic["V"][..., -1], ron_static["V_reset"]),
        )
        self.assertEqual(
            torch.count_nonzero(
                states["synapses"]["Dynamic"]["on_ron"]["dV_gSYN"][..., -1]
            ).item(),
            0,
        )


if __name__ == "__main__":
    unittest.main()
