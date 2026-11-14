"""Verify the probe's gradient intervention and hook/checkpoint boundaries."""

import importlib.util
import unittest
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("M0 tests require torch")

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("shared_value", ROOT / "src/vla_rl_mt/shared_value.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)
        self.last_suffix = None

    def forward(self, embedded):
        self.last_suffix = embedded[1]
        return self.projection(embedded[0])


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = TinyLLM()
        self.img = nn.Identity()
        self.embed_dtype = torch.float32

    def build_prefix_cache(self, observation):
        hidden = self.llm([observation.state[:, None, :], None])
        mask = torch.ones(hidden.shape[:2], dtype=torch.bool)
        return hidden, mask, None


class MethodProbeTests(unittest.TestCase):
    def make_probe(self):
        torch.manual_seed(5)
        probe = shared.SharedPrefixValue(TinyPolicy(), width=4, rank=2, head_width=3)
        self.addCleanup(probe.close)
        return probe

    def observation(self):
        from rlinf.models.embodiment.openpi.modules.model import IMAGE_KEYS, Observation

        return Observation(
            images={key: torch.zeros(1, 224, 224, 3) for key in IMAGE_KEYS},
            image_masks={key: torch.ones(1, dtype=torch.bool) for key in IMAGE_KEYS},
            state=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        )

    def test_zero_adapter_keeps_native_output_and_close_removes_intervention(self):
        torch.manual_seed(5)
        policy = TinyPolicy()
        tokens = torch.randn(1, 2, 4)
        baseline = policy.llm([tokens, None]).detach()
        probe = shared.SharedPrefixValue(policy, width=4, rank=2, head_width=3)
        self.addCleanup(probe.close)
        self.assertTrue(torch.equal(baseline, policy.llm([tokens, None])))
        with torch.no_grad():
            probe.adapter.up.weight.fill_(0.1)
        self.assertFalse(torch.equal(baseline, policy.llm([tokens, None])))
        with self.assertRaisesRegex(ValueError, "already attached"):
            shared.SharedPrefixValue(policy, width=4)
        probe.close()
        self.assertTrue(torch.equal(baseline, policy.llm([tokens, None])))

    def test_value_stop_grad_controls_shared_gradient_without_unfreezing_base(self):
        for detach in (True, False):
            with self.subTest(detach_shared=detach):
                probe = self.make_probe()
                probe.value(self.observation(), detach_shared=detach).square().mean().backward()
                inventory = shared.gradient_inventory(probe)
                self.assertEqual(inventory["frozen_policy"]["gradient_tensors"], 0)
                self.assertGreater(inventory["value_head"]["gradient_l2"], 0)
                self.assertEqual(inventory["adapter"]["gradient_l2"] > 0, not detach)
                self.assertIsNone(probe.policy.llm.last_suffix)

    def test_addon_reload_is_strict_and_does_not_include_frozen_base(self):
        probe = self.make_probe()
        state = probe.addon_state()
        self.assertEqual(set(state), {"adapter", "value_head"})
        with torch.no_grad():
            probe.adapter.up.weight.fill_(1)
        probe.load_addon(state)
        self.assertTrue(torch.equal(probe.adapter.up.weight, state["adapter"]["up.weight"]))
        with self.assertRaises(ValueError):
            probe.load_addon({"adapter": state["adapter"]})
        with self.assertRaises(RuntimeError):
            probe.load_addon({**state, "adapter": {**state["adapter"], "unexpected": torch.zeros(1)}})


if __name__ == "__main__":
    unittest.main()
