"""Bounded frozen-W1 layer probes and sequential-value reference for M4."""

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from vla_rl_mt.shared_value import PrefixAdapter
from vla_rl_mt.value_head import FeatureDataset, PrefixValueHead, task_episode_weights


class LayerValueHead(PrefixValueHead):
    def __init__(self, width, hidden, loss_scale):
        super().__init__(width, hidden)
        self.loss_scale = loss_scale

    def forward(self, observation, target_values=None):
        result = super().forward(observation, target_values)
        if result.loss is not None:
            result.loss = result.loss*self.loss_scale
        return result


def build_model(config, torch_dtype=None):
    with torch.random.fork_rng():
        torch.manual_seed(int(config.init_seed))
        # Match M2 initialization, where the shared adapter was allocated first.
        PrefixAdapter(2048, 32)
        model = LayerValueHead(4096, 63, float(config.loss_scale))
    return model.float()


def register():
    from rlinf.models import register_model

    register_model("mt_layer_value", build_model, force=True)


def make_worker_class():
    from rlinf.data.datasets.recap.value_dataset import ValueDataLoaderImpl
    from rlinf.workers.sft.fsdp_value_sft_worker import FSDPValueSftWorker

    class LayerValueWorker(FSDPValueSftWorker):
        def build_dataloader(self):
            if self._world_size != 1:
                raise ValueError("Layer probes are single-GPU jobs")
            path = Path(self.cfg.data.feature_path)
            cache = torch.load(path, map_location="cpu", weights_only=True)
            if (cache["pool_manifest_sha256"] != self.cfg.data.pool_manifest_sha256
                    or cache["layer"] != int(self.cfg.data.layer)
                    or cache["actor_sha256"] != self.cfg.data.actor_sha256
                    or cache["target_scheme"] != "terminal_binary"
                    or cache["features"].shape[1] != 4096):
                raise ValueError("Layer cache identity mismatch")
            train, val = FeatureDataset(cache, "train"), FeatureDataset(cache, "val")
            count = int(self.cfg.runner.max_steps)*int(self.cfg.actor.global_batch_size)
            seed = int(self.cfg.actor.seed)+202
            weights = task_episode_weights(train.rows)
            draws = torch.multinomial(weights, count, replacement=True, generator=torch.Generator().manual_seed(seed))
            keys = [(train.rows[int(i)]["episode"], train.rows[int(i)]["observation_index"]) for i in draws]
            sampler = WeightedRandomSampler(weights, count, replacement=True, generator=torch.Generator().manual_seed(seed))
            batch = int(self.cfg.actor.micro_batch_size)
            training = DataLoader(train, batch_size=batch, sampler=sampler, drop_last=True, num_workers=0)
            validation = DataLoader(val, batch_size=batch, shuffle=False, num_workers=0)
            metadata = {key: cache[key] for key in ("pool_manifest_sha256", "actor_sha256", "layer", "normalization")}
            metadata.update(feature_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            sample_sequence_sha256=hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
                            sampled_rows=keys, steps=int(self.cfg.runner.max_steps), seed=int(self.cfg.actor.seed),
                            loss_scale=float(self.cfg.actor.model.loss_scale), train_samples=len(train), val_samples=len(val))
            destination = Path(self.cfg.runner.logger.log_path)/self.cfg.runner.logger.experiment_name
            destination.mkdir(parents=True, exist_ok=True)
            (destination/"feature-data.json").write_text(json.dumps(metadata, indent=2)+"\n")
            return ValueDataLoaderImpl(metadata, training), [("development", ValueDataLoaderImpl(metadata, validation))], metadata

    return LayerValueWorker
