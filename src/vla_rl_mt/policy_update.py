"""Native, adapter-only RECAP-style label controls; original RLinf routing.

The pinned native Pi0 FM implementation, official guidance tokenizer and routing
are composed through the official worker loop. This is an explicit research
adaptation, not the unmodified openpi_cfg/full-action-expert RECAP baseline.
"""

import dataclasses
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from vla_rl_mt.joint_value import (
    BoundaryPool,
    digest,
    draw_indices,
    to_observation,
    write_json,
)
from vla_rl_mt.shared_value import SharedPrefixValue

ROOT = Path(__file__).resolve().parents[2]


def guidance_transform(transform):
    from openpi.transforms import TokenizePrompt, compose
    from rlinf.data.datasets.recap.cfg_model import TokenizePromptWithGuidance

    count = sum(isinstance(item, TokenizePrompt) for item in transform.transforms)
    if count != 1:
        raise ValueError("Expected exactly one native prompt tokenizer")
    return compose([TokenizePromptWithGuidance(item.tokenizer, item.discrete_state_input)
                    if isinstance(item, TokenizePrompt) else item for item in transform.transforms])


class GuidedPool(BoundaryPool):
    def __init__(self, *args):
        super().__init__(*args)
        self.transform = guidance_transform(self.transform)


def label_array(pool, path, *, shuffle=False, seed=0):
    report = json.loads(Path(path).read_text())
    mapping = {(row["episode"], row["observation_index"]): row for row in report["rows"] if row["use_for_loss"]}
    if len(mapping) != len(pool.rows):
        raise ValueError("Label coverage differs from the complete frozen decision pool")
    labels = []
    for row in pool.rows:
        source = mapping[row["episode"], row["observation_index"]]
        if any(source[key] != row[key] for key in ("task", "split", "step", "success", "target")):
            raise ValueError("Label identity or original MC targets changed")
        labels.append(bool(source["positive"]))
    labels = torch.tensor(labels, dtype=torch.bool)
    if shuffle:
        generator = torch.Generator().manual_seed(seed+404)
        for task in sorted({row["task"] for row in pool.rows}):
            indices = [i for i, row in enumerate(pool.rows) if row["task"] == task and row["split"] == "train"]
            permutation = torch.randperm(len(indices), generator=generator)
            labels[indices] = labels[torch.tensor(indices)[permutation]].clone()
    return labels


class PolicyPairs(Dataset):
    def __init__(self, pool, *, mode, label_path, steps, batch_size, seed):
        self.pool, self.mode, self.seed, self.offset = pool, mode, seed, 0
        success_only = mode in ("bc", "bc_positive")
        candidates = [i for i, row in enumerate(pool.rows) if row["split"] == "train"
                      and (not success_only or row["success"])]
        self.draws = draw_indices(pool.rows, candidates, steps*batch_size, seed+(101 if success_only else 303))
        self.labels = torch.ones(len(pool.rows), dtype=torch.bool) if success_only else label_array(pool, label_path)
        if mode == "shuffle_cond":
            # Preserve both dataset label ratios and actual scheduled positive
            # exposure, including guidance dropout; don't confound randomization
            # with a different amount of conditional training.
            counts = Counter(int(i) for i in self.draws)
            eligible = Counter()
            for index, row in enumerate(self.draws):
                generator = torch.Generator().manual_seed(seed+2_000_000+index)
                torch.randn((32, 32), generator=generator)
                torch.rand((), generator=generator)
                if float(torch.rand((), generator=generator)) > 0.1:
                    eligible[int(row)] += 1
            strata = defaultdict(list)
            for index, row in enumerate(pool.rows):
                if row["split"] == "train":
                    strata[row["task"], counts[index], eligible[index]].append(index)
            generator = torch.Generator().manual_seed(seed+404)
            for indices in strata.values():
                permutation = torch.randperm(len(indices), generator=generator)
                self.labels[indices] = self.labels[torch.tensor(indices)[permutation]].clone()

    def __len__(self):
        return len(self.draws)

    def __getitem__(self, index):
        index += self.offset
        row = int(self.draws[index])
        raw, actions, mask = self.pool.sample(row)
        generator = torch.Generator().manual_seed(self.seed+2_000_000+index)
        noise = torch.randn(actions.shape, generator=generator)
        flow_time = torch.rand((), generator=generator).pow(1/1.5)*0.999+0.001
        routing_random = torch.rand((), generator=generator)
        return {"native": raw, "actions": actions, "action_mask": mask, "noise": noise,
                "flow_time": flow_time, "routing_random": routing_random, "positive": self.labels[row],
                "success": torch.tensor(self.pool.rows[row]["success"])}


def masked_per_sample(loss, mask):
    if loss.shape != mask.shape or mask.dtype != torch.bool or not bool(mask.sum((1, 2)).gt(0).all()):
        raise ValueError("Invalid real-action mask")
    return loss.masked_fill(~mask, 0).sum((1, 2))/mask.sum((1, 2))


class NativePolicyUpdate(SharedPrefixValue):
    def __init__(self, policy, mode, **kwargs):
        if mode not in ("bc", "bc_positive", "c1_cond", "c2_cond", "shuffle_cond", "c1_weighted", "c2_weighted",
                        "last_weighted", "layer_weighted"):
            raise ValueError(f"Unknown M4 control: {mode}")
        super().__init__(policy, register_policy=False, **kwargs)
        self.mode = mode
        self.value_head.requires_grad_(False)
        self.step_metrics = []
        self.exposure = {name: 0 for name in ("samples", "success", "positive", "positive_conditional")}

    def forward(self, observation, target_values=None):
        from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
            compute_cfg_routing_masks,
        )
        from rlinf.models.embodiment.value_model.recap.modeling_critic import (
            CriticOutput,
        )

        data = observation
        routing = compute_cfg_routing_masks(data["positive"], positive_only_conditional=True,
                                            unconditional_prob=0.1, random_values=data["routing_random"])
        keep = routing["positive_conditional_mask"]
        obs = data["obs"]
        if self.mode.endswith("cond") or self.mode == "bc_positive":
            obs = dataclasses.replace(obs,
                tokenized_prompt=torch.where(keep[:, None], data["positive_tokens"], obs.tokenized_prompt),
                tokenized_prompt_mask=torch.where(keep[:, None], data["positive_mask"], obs.tokenized_prompt_mask))
        elementwise = self.policy.compute_loss(obs, data["actions"], train=False,
                                               noise=data["noise"], time=data["flow_time"])
        per_sample = masked_per_sample(elementwise, data["action_mask"])
        if self.mode.endswith("weighted"):
            # Match conditional-positive exposure and its contribution per
            # global batch; don't rescale by random positive counts.
            per_sample = per_sample*keep.float()
        loss = per_sample.mean()
        self.step_metrics.append({"sft_loss": float(loss.detach()), "value_loss": 0.0})
        self.exposure["samples"] += len(keep)
        self.exposure["success"] += int(data["success"].sum())
        self.exposure["positive"] += int(data["positive"].sum())
        self.exposure["positive_conditional"] += int(keep.sum())
        return CriticOutput(loss=loss)


def build_model(config, torch_dtype=None):
    sys.path.insert(0, str(ROOT/"scripts"))
    from evaluate_multitask import load_model

    plan = json.loads((ROOT/"configs/multitask/method_m4.json").read_text())
    manifest = json.loads((Path(config.pool_path)/"manifest.json").read_text())
    parent_path = ROOT/plan["w1_checkpoint"]
    if (digest(parent_path) != plan["parent_checkpoint_sha256"]["w1"]
            or digest(ROOT/manifest["protocol"]["model"]["path"]/"model.safetensors")
                != manifest["protocol"]["model"]["sha256"]):
        raise ValueError("M4 pinned parent/base changed")
    parent = torch.load(parent_path, weights_only=True, map_location="cpu")
    policy, _ = load_model("native", manifest["protocol"])
    policy.gradient_checkpointing_enable({"use_reentrant": False})
    model = NativePolicyUpdate(policy, str(config.mode), **parent["architecture"]).to("cuda").train()
    model.load_addon(parent["addon"])
    return model


def register():
    from rlinf.models import register_model

    register_model("mt_native_policy", build_model, force=True)


def make_worker_class():
    from rlinf.data.datasets.recap.value_dataset import ValueDataLoaderImpl
    from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager

    from vla_rl_mt.joint_value import make_worker_class as joint_worker

    class NativePolicyWorker(joint_worker()):
        def build_dataloader(self):
            self.pool = GuidedPool(self.cfg.data.pool_path, self.cfg.data.pool_manifest_sha256)
            if (self._world_size != 1 or not self.cfg.actor.fsdp_config.use_orig_params
                    or self.cfg.actor.fsdp_config.sharding_strategy != "no_shard"):
                raise ValueError("M4 addon training requires single-GPU NO_SHARD/original parameters")
            if (self.cfg.data.label_path != "none"
                    and digest(self.cfg.data.label_path) != self.cfg.data.label_sha256):
                raise ValueError("Frozen label file checksum changed")
            self.pairs = PolicyPairs(self.pool, mode=str(self.cfg.actor.model.mode), label_path=str(self.cfg.data.label_path),
                steps=int(self.cfg.runner.max_steps), batch_size=int(self.cfg.actor.global_batch_size), seed=int(self.cfg.actor.seed))
            loader = DataLoader(self.pairs, batch_size=int(self.cfg.actor.micro_batch_size), num_workers=0)
            self.destination = Path(self.cfg.runner.logger.log_path)/self.cfg.runner.logger.experiment_name
            self.destination.mkdir(parents=True, exist_ok=True)
            plan = json.loads((ROOT/"configs/multitask/method_m4.json").read_text())
            keys = [(self.pool.rows[int(i)]["episode"], self.pool.rows[int(i)]["observation_index"]) for i in self.pairs.draws]
            self.identity = {"scope": "native adapter-only RECAP-style control, official tokenizer/routing/worker loop",
                "mode": str(self.cfg.actor.model.mode), "pool_manifest_sha256": str(self.cfg.data.pool_manifest_sha256),
                "parent_sha256": plan["parent_checkpoint_sha256"]["w1"],
                "sample_sequence_sha256": hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
                "label_sha256": digest(self.cfg.data.label_path) if self.cfg.data.label_path != "none" else None,
                "source_sha256": digest(Path(__file__)), "steps": int(self.cfg.runner.max_steps),
                "global_batch": int(self.cfg.actor.global_batch_size), "micro_batch": int(self.cfg.actor.micro_batch_size),
                "lr": float(self.cfg.actor.optim.lr), "shuffle_within_train_task": self.cfg.actor.model.mode == "shuffle_cond",
                "deploy_prompt": "positive" if str(self.cfg.actor.model.mode).endswith("cond")
                    or self.cfg.actor.model.mode == "bc_positive" else "plain"}
            write_json(self.destination/"live-data.json", self.identity)
            self._first_step = True
            return ValueDataLoaderImpl(self.identity, loader), [], self.identity

        def _prepare_input(self, batch):
            raw = batch["native"]
            data = {key: value.to(self.device) for key, value in batch.items() if key != "native"}
            data.update(obs=to_observation(raw, self.device),
                        positive_tokens=raw["tokenized_positive_guidance_prompt"].to(self.device),
                        positive_mask=raw["tokenized_positive_guidance_prompt_mask"].to(self.device))
            return data, None, {}

        def run_eval(self):
            with self.worker_timer():
                return {}

        def save_checkpoint(self, save_path, step=0):
            FSDPModelManager.save_checkpoint(self, save_path, step)
            model = self.model.module
            torch.save({"kind": "M4 native policy addon", "identity": self.identity,
                        "architecture": model.architecture, "step": step, "addon": model.addon_state(),
                        "exposure": model.exposure}, Path(save_path)/"addon.pt")
            write_json(Path(save_path)/"identity.json", self.identity)
            write_json(self.destination/"exposure.json", model.exposure)

    return NativePolicyWorker
