import copy
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from Simulation import (
    Architecture_Declaration,
    Loss_handler,
    Reset_handler,
    conditional_handler,
    ode_handler,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64


def central_difference(function, value, epsilon=1e-6):
    plus = function(torch.tensor(value + epsilon, dtype=DTYPE))
    minus = function(torch.tensor(value - epsilon, dtype=DTYPE))
    return (plus - minus) / (2.0 * epsilon)


def direct_gsyn_objective(gsyn, return_trace=False):
    """Smooth voltage objective with a prescribed, parameter-independent PSC."""
    dt = 0.1
    neuron = {"R": 200.0, "tau": 20.0, "E_k": -80.0}
    synapse = {"ESYN": 0.0}
    e_l = -65.0
    threshold = -47.0
    width = 3.0
    voltage = torch.tensor(-49.0, dtype=DTYPE)
    d_voltage = torch.tensor(0.0, dtype=DTYPE)
    objective = torch.tensor(0.0, dtype=DTYPE)
    objective_derivative = torch.tensor(0.0, dtype=DTYPE)
    psc_values = torch.tensor(
        [0.020, 0.035, 0.050, 0.045, 0.030, 0.015, 0.040, 0.025],
        dtype=DTYPE,
    )
    adaptation_values = torch.tensor(
        [0.0008, 0.0009, 0.0010, 0.0011, 0.0010, 0.0009, 0.0008, 0.0007],
        dtype=DTYPE,
    )

    for psc_s, g_ad in zip(psc_values, adaptation_values):
        old_voltage = voltage
        voltage_jacobian = (
            -1.0 - neuron["R"] * g_ad - neuron["R"] * gsyn * psc_s
        ) / neuron["tau"]
        direct_source = ode_handler.conductance_voltage_source(
            neuron, synapse, psc_s, old_voltage
        )
        d_voltage = ode_handler.advance_voltage_sensitivity(
            d_voltage, voltage_jacobian, direct_source, dt
        )

        voltage_rhs = (
            (e_l - old_voltage)
            - neuron["R"] * g_ad * (old_voltage - neuron["E_k"])
            - neuron["R"] * gsyn * psc_s * (old_voltage - synapse["ESYN"])
        ) / neuron["tau"]
        voltage = old_voltage + dt * voltage_rhs

        offset = (voltage - threshold) / width
        objective = objective + torch.tanh(offset)
        psi = (1.0 - torch.tanh(offset) ** 2) / width
        objective_derivative = objective_derivative + psi * d_voltage

    if return_trace:
        return objective, objective_derivative, voltage, d_voltage
    return objective


def output_ad_objective(output_ad, return_trace=False):
    """Frozen-event output-adaptation objective matching the manual recursion."""
    dt = 0.1
    neuron = {"R": 200.0, "tau": 20.0, "E_k": -80.0}
    e_l = -65.0
    tau_ad = 100.0
    threshold = -47.0
    reset = -54.0
    width = 3.0
    voltage = torch.tensor(-48.5, dtype=DTYPE)
    g_ad = torch.tensor(0.001, dtype=DTYPE)
    d_voltage = torch.tensor(0.0, dtype=DTYPE)
    d_g_ad = torch.tensor(0.0, dtype=DTYPE)
    objective = torch.tensor(0.0, dtype=DTYPE)
    objective_derivative = torch.tensor(0.0, dtype=DTYPE)
    contributions = []
    accepted_spikes = (True, False, False, True, False, False, False, False)

    for accepted_spike in accepted_spikes:
        old_voltage = voltage
        old_g_ad = g_ad
        voltage_jacobian = (
            -1.0 - neuron["R"] * old_g_ad
        ) / neuron["tau"]
        d_voltage_free = ode_handler.advance_voltage_sensitivity(
            d_voltage,
            voltage_jacobian,
            ode_handler.adaptation_voltage_source(
                neuron, old_voltage, d_g_ad
            ),
            dt,
        )
        d_g_ad_decay = ode_handler.advance_decay_sensitivity(
            d_g_ad, tau_ad, dt
        )

        voltage_rhs = (
            (e_l - old_voltage)
            - neuron["R"] * old_g_ad * (old_voltage - neuron["E_k"])
        ) / neuron["tau"]
        voltage_free = old_voltage + dt * voltage_rhs
        g_ad_decay = old_g_ad + dt * (-old_g_ad / tau_ad)

        offset = (voltage_free - threshold) / width
        objective = objective + torch.tanh(offset)
        psi = (1.0 - torch.tanh(offset) ** 2) / width
        contribution = psi * d_voltage_free
        contributions.append(contribution)
        objective_derivative = objective_derivative + contribution

        if accepted_spike:
            voltage = torch.tensor(reset, dtype=DTYPE)
            g_ad = g_ad_decay + output_ad
            d_voltage = torch.tensor(0.0, dtype=DTYPE)
            d_g_ad = d_g_ad_decay + 1.0
        else:
            voltage = voltage_free
            g_ad = g_ad_decay
            d_voltage = d_voltage_free
            d_g_ad = d_g_ad_decay

    if return_trace:
        return (
            objective,
            objective_derivative,
            voltage,
            d_voltage,
            torch.stack(contributions),
        )
    return objective


def input_strf_objective(parameter, rate_direction, steps=None, return_trace=False):
    """No-event input-neuron objective with a smooth Bernoulli-drive tangent."""
    dt = 0.1
    neuron = {
        "R": 200.0,
        "tau": 20.0,
        "E_k": -80.0,
        "g_postIC": 0.17,
        "E_exc": 0.0,
    }
    e_l = -65.0
    tau_ad = 100.0
    threshold = -47.0
    width = 3.0
    voltage = torch.tensor(-55.0, dtype=DTYPE)
    g_ad = torch.tensor(0.0008, dtype=DTYPE)
    d_voltage = torch.tensor(0.0, dtype=DTYPE)
    objective = torch.tensor(0.0, dtype=DTYPE)
    objective_derivative = torch.tensor(0.0, dtype=DTYPE)
    sampled_inputs = torch.tensor(
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        dtype=DTYPE,
    )
    rate_direction = torch.as_tensor(rate_direction, dtype=DTYPE)
    if steps is not None:
        sampled_inputs = sampled_inputs[:steps]
        rate_direction = rate_direction[:steps]

    for sampled_input, current_rate_derivative in zip(sampled_inputs, rate_direction):
        old_voltage = voltage
        old_g_ad = g_ad
        base_input = sampled_input
        smooth_input = (
            base_input
            + (dt / 1000.0) * current_rate_derivative * parameter
        )

        voltage_jacobian = ode_handler.input_voltage_jacobian(
            neuron, old_g_ad, base_input
        )
        source = ode_handler.input_rate_voltage_source(
            neuron, old_voltage, current_rate_derivative, dt
        )
        d_voltage = ode_handler.advance_voltage_sensitivity(
            d_voltage, voltage_jacobian, source, dt
        )

        voltage_rhs = (
            (e_l - old_voltage)
            - neuron["R"] * old_g_ad * (old_voltage - neuron["E_k"])
            - neuron["R"] * neuron["g_postIC"] * smooth_input
            * (old_voltage - neuron["E_exc"])
        ) / neuron["tau"]
        voltage = old_voltage + dt * voltage_rhs
        g_ad = old_g_ad + dt * (-old_g_ad / tau_ad)

        offset = (voltage - threshold) / width
        objective = objective + torch.tanh(offset)
        psi = (1.0 - torch.tanh(offset) ** 2) / width
        objective_derivative = objective_derivative + psi * d_voltage

    if return_trace:
        return objective, objective_derivative, voltage, d_voltage
    return objective


def exact_stp_flow(state, duration, static):
    """Exact between-event flow for [F, P, q, x, s]."""
    F, P, q, x, s = np.asarray(state, dtype=np.float64)
    decay_F = np.exp(-duration / static["tauF"])
    decay_P = np.exp(-duration / static["tauP"])
    decay_D = np.exp(-duration / static["tauD"])
    decay_R = np.exp(-duration / static["tauR"])
    s_from_x = (
        static["scale"]
        * x
        * static["tauD"]
        / (static["tauD"] - static["tauR"])
        * (decay_D - decay_R)
    )
    return np.array(
        [
            1.0 + (F - 1.0) * decay_F,
            1.0 + (P - 1.0) * decay_P,
            q,
            x * decay_D,
            s * decay_R + s_from_x,
        ],
        dtype=np.float64,
    )


def exact_stp_tangent_flow(tangent, duration, static):
    """Exact homogeneous fixed-time tangent flow between events."""
    dF, dP, dq, dx, ds = np.asarray(tangent, dtype=np.float64)
    decay_F = np.exp(-duration / static["tauF"])
    decay_P = np.exp(-duration / static["tauP"])
    decay_D = np.exp(-duration / static["tauD"])
    decay_R = np.exp(-duration / static["tauR"])
    ds_from_dx = (
        static["scale"]
        * dx
        * static["tauD"]
        / (static["tauD"] - static["tauR"])
        * (decay_D - decay_R)
    )
    return np.array(
        [dF * decay_F, dP * decay_P, dq, dx * decay_D, ds * decay_R + ds_from_dx],
        dtype=np.float64,
    )


def moving_event_stp_trajectory(theta, event_times, event_directions, static, end_time):
    """Forward STP trajectory with linearly parameterized event times."""
    state = np.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    previous_time = 0.0
    for base_time, direction in zip(event_times, event_directions):
        event_time = base_time + direction * theta
        state = exact_stp_flow(state, event_time - previous_time, static)
        F, P, _, x, s = state
        event_q = F * P
        state = np.array(
            [
                F + static["PSC_fF"] * (static["PSC_maxF"] - F),
                P * (1.0 - static["PSC_fP"]),
                event_q,
                x + event_q,
                s,
            ],
            dtype=np.float64,
        )
        previous_time = event_time
    return exact_stp_flow(state, end_time - previous_time, static)


def network_args():
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


def network_params():
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


class RecursiveEligibilityFiniteDifferenceTests(unittest.TestCase):
    def test_direct_gsyn_recursive_trace_matches_float64_finite_difference(self):
        base_value = 0.03
        _, analytic, _, _ = direct_gsyn_objective(
            torch.tensor(base_value, dtype=DTYPE), return_trace=True
        )
        finite_difference = central_difference(direct_gsyn_objective, base_value)
        torch.testing.assert_close(
            analytic, finite_difference, rtol=2e-8, atol=2e-9
        )

    def test_output_ad_recursive_trace_matches_float64_finite_difference(self):
        base_value = 0.004
        _, analytic, _, _, contributions = output_ad_objective(
            torch.tensor(base_value, dtype=DTYPE), return_trace=True
        )
        finite_difference = central_difference(output_ad_objective, base_value)
        torch.testing.assert_close(
            analytic, finite_difference, rtol=2e-8, atol=2e-9
        )
        # The increment is applied after the accepted spike, so that spike
        # cannot receive eligibility from its own adaptation increment.
        self.assertEqual(float(contributions[0]), 0.0)

    def test_input_strf_one_step_trace_matches_float64_finite_difference(self):
        rate_direction = torch.tensor(
            [850.0, -400.0, 300.0, 0.0, 600.0, -250.0, 500.0, 200.0, -100.0, 450.0],
            dtype=DTYPE,
        )
        function = lambda value: input_strf_objective(
            value, rate_direction, steps=1
        )
        _, analytic, _, _ = input_strf_objective(
            torch.tensor(0.0, dtype=DTYPE),
            rate_direction,
            steps=1,
            return_trace=True,
        )
        finite_difference = central_difference(function, 0.0, epsilon=3e-5)
        torch.testing.assert_close(
            analytic, finite_difference, rtol=2e-8, atol=2e-9
        )

    def test_input_strf_multistep_gain_and_alpha_traces_match_float64_fd(self):
        directions = {
            "gain": torch.tensor(
                [850.0, -400.0, 300.0, 0.0, 600.0, -250.0, 500.0, 200.0, -100.0, 450.0],
                dtype=DTYPE,
            ),
            "alpha": torch.tensor(
                [-3.0, 2.0, 4.0, -1.0, 0.5, 3.0, -2.0, 1.5, 2.5, -0.5],
                dtype=DTYPE,
            ),
        }
        for param_name, rate_direction in directions.items():
            with self.subTest(param_name=param_name):
                function = lambda value: input_strf_objective(
                    value, rate_direction
                )
                _, analytic, _, _ = input_strf_objective(
                    torch.tensor(0.0, dtype=DTYPE),
                    rate_direction,
                    return_trace=True,
                )
                finite_difference = central_difference(function, 0.0, epsilon=3e-5)
                torch.testing.assert_close(
                    analytic, finite_difference, rtol=3e-8, atol=3e-9
                )

    def test_completed_bins_have_exact_nonoverlapping_membership(self):
        args = network_args()
        spikes = torch.zeros((1, 10, 1, 6), dtype=torch.int64)
        spikes[0, 0, 0, 0] = 1
        spikes[0, 0, 0, 2] = 1
        spikes[0, 0, 0, 3] = 1
        states = {
            "neurons": {
                "Dynamic": {
                    "ron": {
                        "spikes_holder": spikes,
                        "mean_sse_loss": 0,
                    }
                }
            }
        }
        target = torch.tensor([[2.0, 1.0]])
        first = Loss_handler.calculate_loss(states, target, args, 2, 3)
        second = Loss_handler.calculate_loss(states, target, args, 5, 3)
        self.assertEqual(float(first["loss"]), 0.0)
        self.assertEqual(float(second["loss"]), 0.0)

    def test_sonoff_spike_wires_upstream_gsyn_into_indirect_delay(self):
        args = network_args()
        states = Architecture_Declaration.build_network(
            args, torch.device("cpu"), network_params()
        )
        sonoff = states["neurons"]["Dynamic"]["sonoff"]
        sonoff["V"][..., -1].fill_(-46.0)
        sonoff["V_slope"] = torch.full_like(sonoff["V"][..., -1], 2.0)
        states["synapses"]["Dynamic"]["on_sonoff"]["dV_gSYN"][..., -1].fill_(4.0)

        conditional_handler.condtion2(args, states, timestep=0)

        delay_buffer = states["synapses"]["Dynamic"]["sonoff_ron"][
            "dtime_on_sonoff_gSYN_delay"
        ]
        expected = -4.0 / 2.0
        self.assertAlmostEqual(float(delay_buffer[0, 0, 0, 0]), expected, places=6)
        self.assertEqual(
            float(states["synapses"]["Dynamic"]["on_sonoff"]["dV_gSYN"][0, 0, 0, -1]),
            0.0,
        )

    def test_soft_event_strength_stp_map_matches_float64_finite_difference(self):
        static = {
            "PSC_fF": 0.25,
            "PSC_fP": 0.35,
            "PSC_maxF": 4.0,
            "tauF": 180.0,
            "tauP": 30.0,
            "tauD": 1.5,
            "tauR": 0.7,
            "scale": (1.5 / 0.7) ** (0.7 / (1.5 - 0.7)),
        }
        base = np.array([1.3, 0.65, 0.2, -0.1, 0.4, -0.3], dtype=np.float64)
        direction = np.array([0.2, -0.15, 0.05, 0.07, -0.03, 0.11], dtype=np.float64)

        for hard_event in (0.0, 1.0):
            with self.subTest(hard_event=hard_event):
                F, P, x, s, event_strength, _ = base
                dF, dP, dx, ds, w, _ = direction
                tangent = conditional_handler.soft_event_stp_tangent(
                    torch.tensor(F, dtype=DTYPE),
                    torch.tensor(P, dtype=DTYPE),
                    torch.tensor(dF, dtype=DTYPE),
                    torch.tensor(dP, dtype=DTYPE),
                    torch.tensor(dx, dtype=DTYPE),
                    torch.tensor(ds, dtype=DTYPE),
                    torch.tensor(hard_event, dtype=torch.bool),
                    torch.tensor(w, dtype=DTYPE),
                    static,
                )
                analytic = np.array([float(value) for value in tangent[:4]])

                def relaxed_map(theta):
                    F_t = F + theta * dF
                    P_t = P + theta * dP
                    x_t = x + theta * dx
                    s_t = s + theta * ds
                    u_t = hard_event + theta * w
                    event_q = F_t * P_t
                    return np.array(
                        [
                            F_t + u_t * static["PSC_fF"] * (static["PSC_maxF"] - F_t),
                            P_t * (1.0 - u_t * static["PSC_fP"]),
                            x_t + u_t * event_q,
                            s_t,
                        ]
                    )

                epsilon = 1e-6
                finite_difference = (
                    relaxed_map(epsilon) - relaxed_map(-epsilon)
                ) / (2.0 * epsilon)
                np.testing.assert_allclose(
                    analytic, finite_difference, rtol=2e-9, atol=2e-10
                )

    def test_multi_event_depression_and_s_saltation_match_finite_difference(self):
        static = {
            "PSC_fF": 0.25,
            "PSC_fP": 0.35,
            "PSC_maxF": 4.0,
            "tauF": 180.0,
            "tauP": 30.0,
            "tauD": 1.5,
            "tauR": 0.7,
            "scale": (1.5 / 0.7) ** (0.7 / (1.5 - 0.7)),
        }
        event_times = np.array([5.0, 12.0, 20.0])
        event_directions = np.array([0.7, -0.4, 1.1])
        end_time = 35.0

        state = np.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64)
        tangent = np.zeros(5, dtype=np.float64)
        previous_time = 0.0
        for event_time, zeta in zip(event_times, event_directions):
            duration = event_time - previous_time
            state = exact_stp_flow(state, duration, static)
            tangent = exact_stp_tangent_flow(tangent, duration, static)
            F, P, _, x, s = state
            result = conditional_handler.time_shift_stp_tangent(
                torch.tensor(F, dtype=DTYPE),
                torch.tensor(P, dtype=DTYPE),
                torch.tensor(tangent[0], dtype=DTYPE),
                torch.tensor(tangent[1], dtype=DTYPE),
                torch.tensor(tangent[3], dtype=DTYPE),
                torch.tensor(tangent[4], dtype=DTYPE),
                torch.tensor(True),
                torch.tensor(zeta, dtype=DTYPE),
                static,
            )
            dF, dP, dx, ds, d_event_q = [float(value) for value in result]
            tangent = np.array([dF, dP, d_event_q, dx, ds])
            event_q = F * P
            state = np.array(
                [
                    F + static["PSC_fF"] * (static["PSC_maxF"] - F),
                    P * (1.0 - static["PSC_fP"]),
                    event_q,
                    x + event_q,
                    s,
                ]
            )
            previous_time = event_time

        state = exact_stp_flow(state, end_time - previous_time, static)
        analytic = exact_stp_tangent_flow(
            tangent, end_time - previous_time, static
        )
        epsilon = 1e-5
        finite_difference = (
            moving_event_stp_trajectory(
                epsilon, event_times, event_directions, static, end_time
            )
            - moving_event_stp_trajectory(
                -epsilon, event_times, event_directions, static, end_time
            )
        ) / (2.0 * epsilon)
        np.testing.assert_allclose(
            analytic, finite_difference, rtol=2e-7, atol=2e-9
        )
        self.assertNotEqual(float(analytic[1]), 0.0)
        self.assertNotEqual(float(analytic[4]), 0.0)

    def test_sonoff_time_shift_delivery_updates_p_x_s_without_eligibility(self):
        args = network_args()
        states = Architecture_Declaration.build_network(
            args, torch.device("cpu"), network_params()
        )
        sonoff = states["neurons"]["Dynamic"]["sonoff"]
        sonoff["V"][..., -1].fill_(-46.0)
        sonoff["V_slope"] = torch.full_like(sonoff["V"][..., -1], 2.0)
        states["synapses"]["Dynamic"]["on_sonoff"]["dV_gSYN"][..., -1].fill_(4.0)

        conditional_handler.condtion2(args, states, timestep=0)
        delay = int(
            states["synapses"]["Static"]["sonoff_ron"]["PSC_delay"]
            / args["simulation"]["dt"]
        )
        conditional_handler.condtion3(args, states, timestep=0)
        for timestep in range(1, delay + 1):
            sonoff["spike_mask"].zero_()
            conditional_handler.condtion3(args, states, timestep=timestep)

        dynamic = states["synapses"]["Dynamic"]["sonoff_ron"]
        static = states["synapses"]["Static"]["sonoff_ron"]
        zeta = -2.0
        self.assertAlmostEqual(
            float(dynamic["dPSC_P_on_sonoff_gSYN"][0, 0, 0, -1]),
            -static["PSC_fP"] * zeta / static["tauP"],
            places=6,
        )
        self.assertAlmostEqual(
            float(dynamic["dPSC_x_on_sonoff_gSYN"][0, 0, 0, -1]),
            zeta / static["tauD"],
            places=6,
        )
        self.assertAlmostEqual(
            float(dynamic["dPSC_s_on_sonoff_gSYN"][0, 0, 0, -1]),
            -static["scale"] * zeta / static["tauR"],
            places=6,
        )
        self.assertAlmostEqual(float(dynamic["PSC_x"][0, 0, 0, -1]), 1.0)

    def test_strf_soft_delivery_updates_stp_tangent_at_forward_event(self):
        args = network_args()
        states = Architecture_Declaration.build_network(
            args, torch.device("cpu"), network_params()
        )
        dynamic = states["synapses"]["Dynamic"]["on_ron"]
        static = states["synapses"]["Static"]["on_ron"]
        onset = states["neurons"]["Dynamic"]["on"]
        dynamic["dsoft_gain_delay"][0, 0, 0, 0] = 0.3
        delay = int(static["PSC_delay"] / args["simulation"]["dt"])

        onset["spike_mask"].fill_(True)
        conditional_handler.condtion3(args, states, timestep=0)
        for timestep in range(1, delay + 1):
            onset["spike_mask"].zero_()
            conditional_handler.condtion3(args, states, timestep=timestep)

        self.assertAlmostEqual(
            float(dynamic["dPSC_P_gain"][0, 0, 0, -1]), -0.03, places=6
        )
        self.assertAlmostEqual(
            float(dynamic["dPSC_x_gain"][0, 0, 0, -1]), 0.3, places=6
        )
        self.assertEqual(float(dynamic["dPSC_s_gain"][0, 0, 0, -1]), 0.0)
        self.assertAlmostEqual(float(dynamic["PSC_x"][0, 0, 0, -1]), 1.0)

    def test_epoch_reset_clears_stp_tangents_and_separated_buffers(self):
        args = network_args()
        states = Architecture_Declaration.build_network(
            args, torch.device("cpu"), network_params()
        )
        for dynamic in states["neurons"]["Dynamic"].values():
            dynamic["spike_mask"].fill_(True)
        for dynamic in states["synapses"]["Dynamic"].values():
            dynamic["spike_delay_queue"].fill_(True)
            for key, value in dynamic.items():
                if key.startswith(("dPSC_F_", "dPSC_P_", "dPSC_x_", "dPSC_s_", "dsoft_", "dtime_")):
                    value.fill_(1.0)

        Reset_handler.reset_dyanmics(states, args, torch.device("cpu"))

        checked = 0
        for dynamic in states["neurons"]["Dynamic"].values():
            self.assertEqual(int(torch.count_nonzero(dynamic["spike_mask"])), 0)
        for dynamic in states["synapses"]["Dynamic"].values():
            self.assertEqual(int(torch.count_nonzero(dynamic["spike_delay_queue"])), 0)
            for key, value in dynamic.items():
                if key.startswith(("dPSC_F_", "dPSC_P_", "dPSC_x_", "dPSC_s_", "dsoft_", "dtime_")):
                    self.assertEqual(int(torch.count_nonzero(value)), 0, key)
                    checked += 1
        self.assertGreater(checked, 0)

    def test_run_odes_uses_old_psc_and_voltage_for_direct_gsyn_source(self):
        args = network_args()
        states = Architecture_Declaration.build_network(
            args, torch.device("cpu"), network_params()
        )
        ron = states["neurons"]["Dynamic"]["ron"]
        on_ron = states["synapses"]["Dynamic"]["on_ron"]
        ron["V"][..., -1].fill_(-48.0)
        ron["g_ad"][..., -1].zero_()
        ron["noise_sn"][..., -1].zero_()
        on_ron["PSC_s"][..., -1].fill_(0.05)
        # A large x makes s_new very different; the voltage and its
        # sensitivity must nevertheless use s_old=0.05.
        on_ron["PSC_x"][..., -1].fill_(1.0)
        spikes = {
            "onset_offset_spks": {
                "onset_spks": torch.zeros((1, 10, 1, 6)),
                "offset_spks": torch.zeros((1, 10, 1, 6)),
            },
            "noise_spks": torch.zeros((10, 1, 6)),
        }
        static = states["neurons"]["Static"]["ron"]
        syn_static = states["synapses"]["Static"]["on_ron"]
        expected_source = (
            -static["R"] * 0.05 * (-48.0 - syn_static["ESYN"]) / static["tau"]
        )
        expected = args["simulation"]["dt"] * expected_source

        ode_handler.run_odes(args, states, spikes, timestep=0)

        actual = on_ron["dV_gSYN"][0, 0, 0, -1]
        self.assertAlmostEqual(float(actual), expected, places=5)
        self.assertGreater(float(on_ron["PSC_s"][0, 0, 0, -1]), 0.2)

    def test_input_strf_trace_persists_without_a_current_rate_source(self):
        args = network_args()
        states = Architecture_Declaration.build_network(
            args, torch.device("cpu"), network_params()
        )
        spikes = {
            "onset_offset_spks": {
                "onset_spks": torch.zeros((1, 10, 1, 6)),
                "offset_spks": torch.zeros((1, 10, 1, 6)),
            },
            "noise_spks": torch.zeros((10, 1, 6)),
        }
        rate_shape = (6, 1, 1)
        rates = {
            "onset_rate": torch.full(rate_shape, 100.0),
            "offset_rate": torch.zeros(rate_shape),
            "onset_rate_gain_deriv": torch.zeros(rate_shape),
            "offset_rate_gain_deriv": torch.zeros(rate_shape),
            "onset_rate_deriv": torch.zeros(rate_shape),
            "offset_rate_deriv": torch.zeros(rate_shape),
        }
        rates["onset_rate_gain_deriv"][0] = 1000.0

        ode_handler.run_odes(args, states, spikes, timestep=0)
        from Simulation import Eligibility_handler
        Eligibility_handler.update_eligibility(args, states, rates, timestep=0)
        first = states["neurons"]["Dynamic"]["on"]["dV_STRF_gain"][0, 0, 0, -1].clone()

        ode_handler.run_odes(args, states, spikes, timestep=1)
        Eligibility_handler.update_eligibility(args, states, rates, timestep=1)
        second = states["neurons"]["Dynamic"]["on"]["dV_STRF_gain"][0, 0, 0, -1]

        self.assertNotEqual(float(first), 0.0)
        self.assertNotEqual(float(second), 0.0)
        expected = first * (
            1.0 + args["simulation"]["dt"]
            * states["neurons"]["Dynamic"]["on"]["input_V_jacobian"][0, 0, 0]
        )
        torch.testing.assert_close(second, expected, rtol=1e-6, atol=1e-6)

    def test_input_spike_resets_strf_voltage_sensitivities_with_voltage(self):
        args = network_args()
        states = Architecture_Declaration.build_network(
            args, torch.device("cpu"), network_params()
        )
        onset = states["neurons"]["Dynamic"]["on"]
        onset["V"][..., -1].fill_(-46.0)
        onset["dV_STRF_gain"][..., -1].fill_(2.0)
        onset["dV_STRF_alpha"][..., -1].fill_(-3.0)

        conditional_handler.condtion2(args, states, timestep=0)

        self.assertEqual(
            float(onset["V"][0, 0, 0, -1]),
            states["neurons"]["Static"]["on"]["V_reset"],
        )
        self.assertEqual(float(onset["dV_STRF_gain"][0, 0, 0, -1]), 0.0)
        self.assertEqual(float(onset["dV_STRF_alpha"][0, 0, 0, -1]), 0.0)

    def test_mixed_reset_indices_preserve_unselected_sensitivities(self):
        args = network_args()
        states = Architecture_Declaration.build_network(
            args, torch.device("cpu"), network_params()
        )
        onset = states["neurons"]["Dynamic"]["on"]
        mask = torch.zeros_like(onset["V"][..., -1], dtype=torch.bool)
        mask[:, (1, 4, 8), :] = True

        current_before = {}
        for offset, param_name in enumerate(("gain", "alpha"), start=1):
            trace = onset[f"dV_STRF_{param_name}"]
            current_before[param_name] = (
                torch.arange(10, dtype=trace.dtype).reshape(1, 10, 1)
                + 10.0 * offset
            )
            trace[..., -2].fill_(-10.0 * offset)
            trace[..., -1].copy_(current_before[param_name])

        conditional_handler.reset_voltage_sensitivities(
            states, "on", torch.where(mask)
        )

        for offset, param_name in enumerate(("gain", "alpha"), start=1):
            trace = onset[f"dV_STRF_{param_name}"]
            torch.testing.assert_close(
                trace[..., -2][mask], current_before[param_name][mask]
            )
            torch.testing.assert_close(
                trace[..., -1][mask], torch.zeros_like(trace[..., -1][mask])
            )
            torch.testing.assert_close(
                trace[..., -2][~mask],
                torch.full_like(trace[..., -2][~mask], -10.0 * offset),
            )
            torch.testing.assert_close(
                trace[..., -1][~mask], current_before[param_name][~mask]
            )

    def test_refractory_clamp_resets_all_voltage_traces_but_keeps_adaptation_trace(self):
        args = network_args()
        states = Architecture_Declaration.build_network(
            args, torch.device("cpu"), network_params()
        )
        ron = states["neurons"]["Dynamic"]["ron"]
        ron["tspike"][0, 0, 0, 0] = 0
        ron["dV_output_ad"][..., -1].fill_(2.0)
        ron["dg_ad_output_ad"][..., -1].fill_(7.0)
        for syn_name in ("on_ron", "off_ron", "sonoff_ron"):
            dynamic = states["synapses"]["Dynamic"][syn_name]
            dynamic["dV_gain"][..., -1].fill_(2.0)
            dynamic["dV_alpha"][..., -1].fill_(2.0)
            dynamic["dV_gSYN"][..., -1].fill_(2.0)
        indirect = states["synapses"]["Dynamic"]["sonoff_ron"]
        indirect["dV_on_sonoff_gSYN"][..., -1].fill_(2.0)
        indirect["dV_off_sonoff_gSYN"][..., -1].fill_(2.0)

        sonoff = states["neurons"]["Dynamic"]["sonoff"]
        sonoff["tspike"][0, 0, 0, 0] = 0
        for syn_name in ("on_sonoff", "off_sonoff"):
            dynamic = states["synapses"]["Dynamic"][syn_name]
            dynamic["dV_gain"][..., -1].fill_(2.0)
            dynamic["dV_alpha"][..., -1].fill_(2.0)
            dynamic["dV_gSYN"][..., -1].fill_(2.0)

        conditional_handler.condtion1(args, states, timestep=1)

        self.assertEqual(float(ron["dV_output_ad"][0, 0, 0, -1]), 0.0)
        self.assertEqual(float(ron["dg_ad_output_ad"][0, 0, 0, -1]), 7.0)
        for syn_name in ("on_ron", "off_ron", "sonoff_ron"):
            dynamic = states["synapses"]["Dynamic"][syn_name]
            for key in ("dV_gain", "dV_alpha", "dV_gSYN"):
                self.assertEqual(float(dynamic[key][0, 0, 0, -1]), 0.0)
        self.assertEqual(float(indirect["dV_on_sonoff_gSYN"][0, 0, 0, -1]), 0.0)
        self.assertEqual(float(indirect["dV_off_sonoff_gSYN"][0, 0, 0, -1]), 0.0)
        for syn_name in ("on_sonoff", "off_sonoff"):
            dynamic = states["synapses"]["Dynamic"][syn_name]
            for key in ("dV_gain", "dV_alpha", "dV_gSYN"):
                self.assertEqual(float(dynamic[key][0, 0, 0, -1]), 0.0)

    def test_upstream_gsyn_gradient_does_not_require_bk(self):
        args = network_args()
        states = Architecture_Declaration.build_network(
            args, torch.device("cpu"), network_params()
        )
        states["synapses"]["Learnable"]["on_sonoff"]["gSYN_accum"].fill_(2.0)
        Loss_handler.update_grad(states, torch.tensor([[3.0]]))
        self.assertEqual(
            float(states["synapses"]["Learnable"]["on_sonoff"]["gSYN_grad"]),
            6.0,
        )


if __name__ == "__main__":
    unittest.main()
