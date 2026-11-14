"""Exposure-matched shared/independent value heads on one frozen cache."""

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from vla_rl_mt.joint_value import digest, draw_indices, write_json
from vla_rl_mt.value_head import FeatureDataset, build_value_head, make_worker_class as value_worker

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/"configs/multitask/critic_signal.json"


class HeadPairs(Dataset):
    def __init__(self, cache, tasks, selected, steps, seed):
        self.cache = cache
        count = 32//len(selected)
        streams = {}
        for task_index in selected:
            candidates = [i for i, r in enumerate(cache["rows"]) if r["split"] == "train" and r["task"] == tasks[task_index]]
            streams[task_index] = draw_indices(cache["rows"], candidates, 16000, seed+8_000_000+task_index)
        self.draws = torch.tensor([int(streams[task][step*count+i]) for step in range(steps)
                                   for task in selected for i in range(count)])

    def __len__(self):
        return len(self.draws)

    def __getitem__(self, index):
        row = int(self.draws[index])
        return {"observation": self.cache["features"][row],
                "target_values": torch.tensor(self.cache["rows"][row]["target"], dtype=torch.float32)}


def register():
    from rlinf.models import register_model

    register_model("mt_signal_value", build_value_head, force=True)


def make_worker_class():
    from rlinf.data.datasets.recap.value_dataset import ValueDataLoaderImpl

    class SignalValueWorker(value_worker()):
        def build_dataloader(self):
            if self._world_size != 1:
                raise ValueError("Cached signal heads require a single GPU per worker")
            settings = json.loads(PROTOCOL.read_text())
            if digest(PROTOCOL) != self.cfg.data.study_protocol_sha256:
                raise ValueError("Critic-signal protocol changed after planning")
            path = Path(self.cfg.data.feature_path)
            if digest(path) != self.cfg.data.feature_sha256:
                raise ValueError("Frozen feature cache changed")
            cache = torch.load(path, map_location="cpu", weights_only=True)
            if cache["pool_manifest_sha256"] != settings["pool_manifest_sha256"] or cache["target_scheme"] != "terminal_binary":
                raise ValueError("Wrong pool or target semantics")
            group = str(self.cfg.data.head_group)
            selected = settings["head_groups"][group]["tasks"]
            self.pairs = HeadPairs(cache, settings["tasks"], selected, int(self.cfg.runner.max_steps), int(self.cfg.actor.seed))
            train = DataLoader(self.pairs, batch_size=32, num_workers=0)
            keep = [i for i, r in enumerate(cache["rows"]) if r["task"] in [settings["tasks"][k] for k in selected]]
            subset = {**cache, "features": cache["features"][keep], "rows": [cache["rows"][i] for i in keep]}
            val = DataLoader(FeatureDataset(subset, "val"), batch_size=32, num_workers=0)
            self.destination = Path(self.cfg.runner.logger.log_path)/self.cfg.runner.logger.experiment_name
            self.destination.mkdir(parents=True, exist_ok=True)
            self.metadata = {"feature_sha256": digest(path), "pool_manifest_sha256": cache["pool_manifest_sha256"],
                "protocol_sha256": digest(PROTOCOL), "source_sha256": digest(Path(__file__)),
                "group": group, "seed": int(self.cfg.actor.seed), "steps": int(self.cfg.runner.max_steps),
                "hidden": int(self.cfg.actor.model.hidden_width), "draws": len(self.pairs),
                "sample_sequence_sha256": hashlib.sha256(json.dumps(self.pairs.draws.tolist()).encode()).hexdigest(),
                "target_scheme": "terminal_binary", "readout": "vision_prompt_concat", "clock_input": False}
            write_json(self.destination/"feature-data.json", self.metadata)
            write_json(self.destination/"sample-sequence.json", self.pairs.draws.tolist())
            return ValueDataLoaderImpl(self.metadata, train), [("development", ValueDataLoaderImpl(self.metadata, val))], self.metadata

        def save_checkpoint(self, save_path, step=0):
            super().save_checkpoint(save_path, step)
            state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
            torch.save({"metadata": self.metadata, "step": int(step), "state_dict": state}, Path(save_path)/"head.pt")

    return SignalValueWorker
