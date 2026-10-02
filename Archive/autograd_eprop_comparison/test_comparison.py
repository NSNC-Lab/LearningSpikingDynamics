"""Run with: python test_comparison.py --project-root /path/to/repository."""
import argparse
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import yaml

from autograd_model import Model, PARAMETERS, cosine, surrogate

ROOT = None


class SurrogateTests(unittest.TestCase):
    def test_exact_hard_forward_and_tanh_backward(self):
        v = torch.tensor([-54., -47., -43.], requires_grad=True)
        hard = (v >= -47).float()
        z = surrogate(hard, (1 + torch.tanh((v + 47) / 5)) / 2)
        self.assertTrue(torch.equal(z, hard))
        grad, = torch.autograd.grad(z.sum(), v)
        torch.testing.assert_close(grad, (1 - torch.tanh((v + 47) / 5) ** 2) / 10)

    def test_cosine_sign_and_zero(self):
        result = cosine([[1, 0], [1, 0], [1, 0], [0, 0]], [[1, 0], [-1, 0], [0, 1], [1, 1]])
        np.testing.assert_allclose(result[:3], [1, -1, 0])
        self.assertTrue(np.isnan(result[-1]))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from Simulation import Architecture_Declaration, Parameter_initialization
        cls.architecture = Architecture_Declaration
        cls.config = yaml.safe_load((ROOT / "simulation_config.yaml").read_text())
        cls.config["simulation"].update(batch_size=2, cell_targets=[7], device="cpu", sim_len=200, epochs=1)
        cls.config["parameter_initialization"]["from_mat"]["enabled"] = False
        np.random.seed(63)
        cls.params = Parameter_initialization.init_params(cls.config)["params"]

    def make_run(self, reset="detach", batch_slice=slice(None)):
        params = {k: v[batch_slice] for k, v in self.params.items()}
        config = copy.deepcopy(self.config)
        size = params[PARAMETERS[0]].shape[0]
        config["simulation"]["batch_size"] = size
        template = self.architecture.build_network(config, torch.device("cpu"), params)
        p = {k: torch.tensor(params[k], dtype=torch.float32, requires_grad=True) for k in PARAMETERS}
        model = Model(template, p, .1, reset)
        gen = torch.Generator().manual_seed(73)
        inputs = {k: (torch.rand((2, 10, 1, 200), generator=gen) < .2).float()[batch_slice]
                  for k in ("onset_spks", "offset_spks")}
        noise = (torch.rand((2, 10, 1, 200), generator=gen) < .02).float()[batch_slice]
        uniforms = torch.rand((200, 2, 10, 1), generator=gen)[:, batch_slice]
        rates = {k: torch.full((200, size, 1), 100.) for k in
                 ("onset_rate_gain_deriv", "offset_rate_gain_deriv", "onset_rate_deriv", "offset_rate_deriv")}
        emitted = []
        for t in range(200):
            emitted.append(model.step(t, inputs, rates, noise, uniforms[t]))
            if t == 49:
                early_state = model.s["on_ron"]["PSC_P"]
        return p, torch.stack(emitted), early_state

    def test_independent_batch_gradients(self):
        p, output, _ = self.make_run()
        self.assertGreater(float(output.detach().sum()), 0)
        grads = torch.autograd.grad(output[:, 0].sum(), tuple(p.values()))
        for g in grads:
            self.assertTrue(torch.isfinite(g).all())
            self.assertEqual(float(g[1].abs().sum()), 0)
        p1, output1, _ = self.make_run(batch_slice=slice(0, 1))
        self.assertTrue(torch.equal(output[:, :1].detach(), output1.detach()))
        grads1 = torch.autograd.grad(output1.sum(), tuple(p1.values()))
        for batch, single in zip(grads, grads1):
            torch.testing.assert_close(batch[:1], single, rtol=2e-5, atol=2e-5)

    def test_reset_modes_do_not_change_forward(self):
        _, detached, _ = self.make_run("detach")
        p, full, _ = self.make_run("full")
        self.assertTrue(torch.equal(detached.detach(), full.detach()))
        grads = torch.autograd.grad(full.sum(), tuple(p.values()))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_bin_gradients_sum_and_retain_history(self):
        p, output, early_state = self.make_run()
        leaves = tuple(p.values())
        losses = [((part.sum((0, 2, 3)) - 2.5) ** 2).sum() for part in output.split(100)]
        bin_grads = [torch.autograd.grad(loss, leaves, retain_graph=True) for loss in losses]
        cross_bin, = torch.autograd.grad(losses[-1], early_state, retain_graph=True)
        self.assertGreater(float(cross_bin.abs().max()), 0)
        total = torch.autograd.grad(sum(losses), leaves)
        for a, b, g in zip(*bin_grads, total):
            torch.testing.assert_close(a + b, g, rtol=3e-4, atol=3e-4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args, remaining = parser.parse_known_args()
    ROOT = args.project_root.resolve()
    sys.path.insert(0, str(ROOT))
    torch.set_num_threads(1)
    unittest.main(argv=[sys.argv[0], *remaining])
