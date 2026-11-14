"""M1 cached-prefix value head and data adapter for the official value worker."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class PrefixValueHead(nn.Module):
    _no_split_modules = ["Sequential"]

    def __init__(self, width=2048, hidden=128):
        super().__init__()
        self.value_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden),
                                        nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, observation, target_values=None):
        from rlinf.models.embodiment.value_model.recap.modeling_critic import (
            CriticOutput,
        )

        prediction = self.value_head(observation.float()).squeeze(-1)
        loss = None if target_values is None else (prediction-target_values).square().mean()
        mae = None if target_values is None else (prediction-target_values).abs().mean()
        return CriticOutput(loss=loss, predicted_values=prediction, mae=mae)


def build_value_head(config, torch_dtype=None):
    with torch.random.fork_rng():
        torch.manual_seed(int(config.init_seed))
        model = PrefixValueHead(int(config.feature_width), int(config.hidden_width))
    return model.float()


def register():
    from rlinf.models import register_model

    register_model("mt_prefix_value", build_value_head, force=True)


class FeatureDataset(Dataset):
    def __init__(self, cache, split):
        self.indices = [index for index, row in enumerate(cache["rows"])
                        if row["split"] == split and row["use_for_loss"]]
        self.features = cache["features"][self.indices].contiguous()
        self.rows = [cache["rows"][index] for index in self.indices]
        self.targets = torch.tensor([row["target"] for row in self.rows], dtype=torch.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {"observation": self.features[index], "target_values": self.targets[index]}


def task_episode_weights(rows):
    frames = Counter((row["task"], row["episode"]) for row in rows)
    episodes = Counter(task for task, _ in frames)
    return torch.tensor([1/(len(episodes)*episodes[row["task"]]*frames[(row["task"], row["episode"])])
                         for row in rows], dtype=torch.float64)


def make_worker_class():
    from rlinf.data.datasets.recap.value_dataset import ValueDataLoaderImpl
    from rlinf.workers.sft.fsdp_value_sft_worker import FSDPValueSftWorker

    class CachedPrefixValueWorker(FSDPValueSftWorker):
        """Override data plumbing only; training/eval/optimizer/ckpt remain official."""

        def build_dataloader(self):
            if self._world_size != 1:
                raise ValueError("Initial M1 uses separate single-GPU jobs, not distributed sampling")
            path = Path(self.cfg.data.feature_path)
            with path.open("rb") as stream:
                fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
            cache = torch.load(path, map_location="cpu", weights_only=True)
            if cache["pool_manifest_sha256"] != self.cfg.data.pool_manifest_sha256:
                raise ValueError("Feature pool fingerprint differs from locked training config")
            if cache["condition"] != self.cfg.data.condition:
                raise ValueError("Wrong instruction condition cache")
            if cache.get("readout", "global_mean") != self.cfg.data.get("readout", "global_mean"):
                raise ValueError("Wrong prefix readout cache")
            if cache.get("target_scheme", "recap_time_cost") != self.cfg.data.get("target_scheme", "recap_time_cost"):
                raise ValueError("Wrong reward/target scheme cache")
            if cache["features"].shape[1] != int(self.cfg.actor.model.feature_width):
                raise ValueError("Cached feature width differs from the head input")
            train, val = FeatureDataset(cache, "train"), FeatureDataset(cache, "val")
            weights = task_episode_weights(train.rows)
            generator = torch.Generator().manual_seed(int(self.cfg.actor.seed))
            sampler = WeightedRandomSampler(weights,
                                            int(self.cfg.runner.max_steps)*int(self.cfg.actor.global_batch_size),
                                            replacement=True, generator=generator)
            batch = int(self.cfg.actor.micro_batch_size)
            training = DataLoader(train, batch_size=batch, sampler=sampler, drop_last=True, num_workers=0)
            validation = DataLoader(val, batch_size=batch, shuffle=False, num_workers=0)
            metadata = {"feature_sha256": fingerprint, "pool_manifest_sha256": cache["pool_manifest_sha256"],
                        "condition": cache["condition"], "train_samples": len(train), "val_samples": len(val),
                        "train_episodes": len({row["episode"] for row in train.rows}),
                        "val_episodes": len({row["episode"] for row in val.rows}),
                        "sampling": "uniform task / episode / decision",
                        "readout": cache.get("readout", "global_mean"),
                        "target_scheme": cache.get("target_scheme", "recap_time_cost"),
                        "trainable_parameters": (2*int(self.cfg.actor.model.feature_width)
                            + int(self.cfg.actor.model.feature_width)*int(self.cfg.actor.model.hidden_width)
                            + 2*int(self.cfg.actor.model.hidden_width)+1)}
            destination = Path(self.cfg.runner.logger.log_path)/self.cfg.runner.logger.experiment_name
            destination.mkdir(parents=True, exist_ok=True)
            (destination/"feature-data.json").write_text(json.dumps(metadata, indent=2)+"\n")
            return (ValueDataLoaderImpl(metadata, training),
                    [("development", ValueDataLoaderImpl(metadata, validation))], metadata)

    return CachedPrefixValueWorker
