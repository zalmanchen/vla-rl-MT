"""Frozen per-draw critic weights, with identical actor/data/noise interfaces."""

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from vla_rl_mt.joint_value import draw_indices, digest, write_json
from vla_rl_mt.policy_update import GuidedPool
from vla_rl_mt.task_mixing import build_model as original_model, make_worker_class as task_worker

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/"configs/multitask/critic_signal.json"


class CandidatePairs(Dataset):
    def __init__(self, pool, tasks, steps, weights, seed=0):
        self.pool, self.offset = pool, 0
        self.draws, self.noise_keys = [], []
        streams = []
        for task_index, task in enumerate(tasks):
            candidates = [i for i, r in enumerate(pool.rows) if r["split"] == "train" and r["task"] == task]
            streams.append(draw_indices(pool.rows, candidates, 256, seed+9_000_000+task_index))
        for step in range(steps):
            for task_index in range(2):
                for occurrence in range(step*2, step*2+2):
                    self.draws.append(int(streams[task_index][occurrence]))
                    self.noise_keys.append(seed+10_000_000+task_index*10000+occurrence)
        self.draws = torch.tensor(self.draws)
        self.weights = torch.as_tensor(weights, dtype=torch.bool)[:steps*4]
        if len(self.weights) != len(self.draws):
            raise ValueError("Frozen draw weights have incorrect length")

    def __len__(self):
        return len(self.draws)

    def __getitem__(self, index):
        index += self.offset
        row = int(self.draws[index])
        raw, actions, mask = self.pool.sample(row)
        generator = torch.Generator().manual_seed(self.noise_keys[index])
        return {"native": raw, "actions": actions, "action_mask": mask,
                "noise": torch.randn(actions.shape, generator=generator),
                "flow_time": torch.rand((), generator=generator).pow(1/1.5)*.999+.001,
                "routing_random": torch.tensor(.5), "positive": self.weights[index],
                "success": torch.tensor(bool(self.pool.rows[row]["success"]))}


def build_model(config, torch_dtype=None):
    if digest(PROTOCOL) != config.study_protocol_sha256:
        raise ValueError("Frozen signal-policy protocol changed")
    model = original_model(config, torch_dtype)
    model.mode = "c1_weighted"
    return model


def register():
    from rlinf.models import register_model

    register_model("mt_signal_policy", build_model, force=True)


def make_worker_class():
    from rlinf.data.datasets.recap.value_dataset import ValueDataLoaderImpl

    class SignalPolicyWorker(task_worker()):
        def build_dataloader(self):
            # Reuse the established original-Motus identity, transforms and
            # optimizer/export hooks, replacing only the fixed data schedule.
            super().build_dataloader()
            settings = json.loads(PROTOCOL.read_text())
            path = Path(self.cfg.data.draw_weights_path)
            if digest(path) != self.cfg.data.draw_weights_sha256:
                raise ValueError("Frozen draw-weight file changed")
            report = json.loads(path.read_text())
            self.pairs = CandidatePairs(self.pool, settings["tasks"], int(self.cfg.runner.max_steps), report["weights"], settings["actor_seed"])
            actual = [{"episode": self.pool.rows[int(row)]["episode"], "observation_index": self.pool.rows[int(row)]["observation_index"],
                       "noise_key": noise} for row, noise in zip(self.pairs.draws, self.pairs.noise_keys, strict=True)]
            if actual != report["sequence"][:len(actual)]:
                raise ValueError("Worker and sealed candidate sequence differ")
            self.identity.update(scope=settings["scope"], mode="c1_weighted", signal_group=str(self.cfg.actor.model.signal_group),
                study_protocol_sha256=digest(PROTOCOL), signal_policy_source_sha256=digest(Path(__file__)),
                draw_weights_sha256=digest(path), supervision_draws=int(self.pairs.weights.sum()),
                sample_sequence_sha256=hashlib.sha256(json.dumps(actual).encode()).hexdigest(),
                guidance_dropout=0.0, per_draw_weights=True)
            write_json(self.destination/"live-data.json", self.identity)
            write_json(self.destination/"candidate-sequence.json", actual)
            write_json(self.destination/"sample-sequence.json", actual)
            loader = DataLoader(self.pairs, batch_size=int(self.cfg.actor.micro_batch_size), num_workers=0)
            return ValueDataLoaderImpl(self.identity, loader), [], self.identity

    return SignalPolicyWorker
