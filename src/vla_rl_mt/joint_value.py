"""Live boundary-chunk SFT/value experiment using RLinf's value worker loop.

Only single-GPU NO_SHARD addon jobs are supported. The frozen native policy is
deliberately outside FSDP/checkpoints; its fingerprint is required separately.
This continued-SFT mechanism probe is not the original RECAP algorithm.
"""

import copy
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from vla_rl_mt.shared_value import SharedPrefixValue, gradient_inventory, isolated_clip
from vla_rl_mt.value_head import task_episode_weights

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def executed_chunk(actions, horizon=32, model_dim=32):
    """Hold the last executed command for context; mask temporal/dimension pads."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 14 or not 0 < len(actions) <= horizon:
        raise ValueError("Expected 1..32 executed 14-dimensional commands")
    if not np.isfinite(actions).all():
        raise ValueError("Nonfinite executed action")
    padded = np.repeat(actions[-1:], horizon, axis=0)
    padded[:len(actions)] = actions
    mask = np.zeros((horizon, model_dim), dtype=bool)
    mask[:len(actions), :14] = True
    return padded, mask


class BoundaryPool:
    """Read audited PNG observations and actual actions, never frozen features."""

    def __init__(self, path, expected_sha256):
        sys.path.insert(0, str(ROOT/"bootstrap"))
        import h5py
        from cache_method_m1 import validate_pool
        from openpi.transforms import compose
        from rlinf.models.embodiment.openpi.transforms.pipeline import (
            build_openpi_transforms,
        )

        self.path = Path(path).resolve()
        if digest(self.path/"manifest.json") != expected_sha256:
            raise ValueError("Wrong frozen pool manifest")
        self.manifest = json.loads((self.path/"manifest.json").read_text())
        self.audit = validate_pool(self.path, self.manifest)
        protocol = self.manifest["protocol"]
        if digest(ROOT/protocol["model"]["norm_stats"]) != protocol["model"]["norm_stats_sha256"]:
            raise ValueError("Normalization checksum changed")
        transforms, _ = build_openpi_transforms(str(ROOT/protocol["model"]["path"]), "pi05_aloha_robotwin",
            {"norm_stats_path": str(ROOT/protocol["model"]["norm_stats"]),
             "adapt_to_pi": True, "extra_delta_transform": True})
        self.transform = compose(transforms)
        self.rows = []
        for entry in self.manifest["episodes"]:
            with h5py.File(self.path/entry["file"]) as episode:
                steps = episode["observation_steps"][:]
                for index, (left, right) in enumerate(zip(steps[:-1], steps[1:], strict=True)):
                    self.rows.append({"episode": entry["file"], "task": entry["task"], "split": entry["split"],
                                      "seed": entry["seed"], "success": entry["success"],
                                      "observation_index": index, "step": int(left), "next_step": int(right),
                                      "target": float(entry["success"])})
        self.cache = OrderedDict()

    def sample(self, index):
        import cv2
        import h5py

        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        row = self.rows[index]
        with h5py.File(self.path/row["episode"]) as episode:
            images = {}
            for camera in episode["images"]:
                image = cv2.imdecode(episode[f"images/{camera}"][row["observation_index"]], cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Invalid frozen PNG")
                images[camera] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            actions, mask = executed_chunk(episode["actions"][row["step"]:row["next_step"]])
            raw = {"images": images, "state": episode["state"][row["observation_index"]],
                   "prompt": str(episode.attrs["instruction"]), "actions": actions}
        transformed = self.transform(copy.deepcopy(raw))
        actions = torch.as_tensor(np.asarray(transformed.pop("actions"))).float()
        if actions.shape != (32, 32) or not bool(torch.isfinite(actions).all()):
            raise ValueError("Native normalized action shape/value mismatch")
        result = (transformed, actions, torch.from_numpy(mask))
        self.cache[index] = result
        if len(self.cache) > 64:
            self.cache.popitem(last=False)
        return result


def draw_indices(rows, candidates, count, seed):
    weights = task_episode_weights([rows[index] for index in candidates])
    draws = torch.multinomial(weights, count, replacement=True, generator=torch.Generator().manual_seed(seed))
    return torch.tensor(candidates, dtype=torch.int64)[draws]


class JointPairs(Dataset):
    def __init__(self, pool, steps, batch_size, seed):
        self.pool, self.offset = pool, 0
        value = [i for i, row in enumerate(pool.rows) if row["split"] == "train"]
        success = [i for i in value if pool.rows[i]["success"]]
        if not success or not value or any(not any(pool.rows[i]["task"] == task for i in success)
                for task in {pool.rows[i]["task"] for i in value}):
            raise ValueError("Each task requires reliable success SFT episodes")
        self.sft = draw_indices(pool.rows, success, steps*batch_size, seed+101)
        self.value = draw_indices(pool.rows, value, steps*batch_size, seed+202)
        self.seed = seed

    def __len__(self):
        return len(self.sft)

    def __getitem__(self, index):
        index += self.offset
        sft_index, value_index = int(self.sft[index]), int(self.value[index])
        obs, actions, mask = self.pool.sample(sft_index)
        value_obs, _, _ = self.pool.sample(value_index)
        # Separate explicit RNG: value-path changes cannot perturb FM noise/time.
        generator = torch.Generator().manual_seed(self.seed+1_000_000+index)
        noise = torch.randn(actions.shape, generator=generator)
        flow_time = torch.rand((), generator=generator).pow(1/1.5)*0.999+0.001
        return {"observation": {"sft": obs, "value": value_obs, "actions": actions,
                                "action_mask": mask, "noise": noise, "flow_time": flow_time},
                "target_values": torch.tensor(self.pool.rows[value_index]["target"], dtype=torch.float32)}


def to_observation(raw, device):
    from rlinf.models.embodiment.openpi.modules.model import Observation

    def move(value):
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return torch.as_tensor(value).to(device)

    return Observation.from_dict(move(raw))


class JointPrefixModel(SharedPrefixValue):
    def __init__(self, policy, *, mode, value_weight, **kwargs):
        if mode not in ("w0", "w1", "w2"):
            raise ValueError("Expected w0, w1 or w2 gradient mode")
        super().__init__(policy, register_policy=False, **kwargs)
        self.mode, self.value_weight = mode, float(value_weight)
        self.step_metrics = []

    def forward(self, observation, target_values=None):
        from rlinf.models.embodiment.value_model.recap.modeling_critic import (
            CriticOutput,
        )

        sft = self.sft_loss(observation["sft"], observation["actions"], noise=observation["noise"],
                            flow_time=observation["flow_time"], action_mask=observation["action_mask"])
        if self.mode == "w0":
            self.step_metrics.append({"sft_loss": float(sft.detach()), "value_loss": 0.0})
            return CriticOutput(loss=sft)
        prediction = self.value(observation["value"], detach_shared=self.mode == "w1")
        value = (prediction-target_values).square().mean()
        self.step_metrics.append({"sft_loss": float(sft.detach()), "value_loss": float(value.detach())})
        return CriticOutput(loss=sft+self.value_weight*value, predicted_values=prediction,
                            mae=(prediction-target_values).abs().mean())


def build_joint_model(config, torch_dtype=None):
    sys.path.insert(0, str(ROOT/"scripts"))
    from evaluate_multitask import load_model

    manifest = json.loads((Path(config.pool_path)/"manifest.json").read_text())
    protocol = manifest["protocol"]
    if digest(ROOT/protocol["model"]["path"]/"model.safetensors") != protocol["model"]["sha256"]:
        raise ValueError("Pinned base model changed")
    policy, _ = load_model("native", protocol)
    policy.gradient_checkpointing_enable({"use_reentrant": False})
    with torch.random.fork_rng():
        torch.manual_seed(int(config.init_seed))
        model = JointPrefixModel(policy, mode=str(config.mode), value_weight=float(config.value_weight),
                                 rank=int(config.adapter_rank), head_width=int(config.head_width),
                                 readout=str(config.readout))
    return model.to("cuda").train()


def register():
    from rlinf.models import register_model

    register_model("mt_joint_prefix", build_joint_model, force=True)


def make_worker_class():
    from rlinf.data.datasets.recap.value_dataset import ValueDataLoaderImpl
    from rlinf.workers.sft.fsdp_value_sft_worker import FSDPValueSftWorker

    class JointPrefixWorker(FSDPValueSftWorker):
        """Inherit training/AdamW/scheduler/checkpoint loops; narrow routing overrides."""

        def build_dataloader(self):
            if (self._world_size != 1 or self.cfg.actor.fsdp_config.sharding_strategy != "no_shard"
                    or not self.cfg.actor.fsdp_config.use_orig_params):
                raise ValueError("M2 addon isolation requires single-GPU NO_SHARD/use_orig_params")
            self.pool = BoundaryPool(self.cfg.data.pool_path, self.cfg.data.pool_manifest_sha256)
            self.pairs = JointPairs(self.pool, int(self.cfg.runner.max_steps), int(self.cfg.actor.global_batch_size),
                                    int(self.cfg.actor.seed))
            loader = DataLoader(self.pairs, batch_size=int(self.cfg.actor.micro_batch_size), num_workers=0)
            self.destination = Path(self.cfg.runner.logger.log_path)/self.cfg.runner.logger.experiment_name
            self.destination.mkdir(parents=True, exist_ok=True)
            self.identity = {"pool_manifest_sha256": str(self.cfg.data.pool_manifest_sha256),
                             "base_sha256": self.pool.manifest["protocol"]["model"]["sha256"],
                             "norm_sha256": self.pool.manifest["protocol"]["model"]["norm_stats_sha256"],
                             "mode": str(self.cfg.actor.model.mode), "target_scheme": "terminal_binary",
                             "clock_input": False, "live_features": True,
                             "sft_sequence_sha256": hashlib.sha256(self.pairs.sft.numpy().tobytes()).hexdigest(),
                             "value_sequence_sha256": hashlib.sha256(self.pairs.value.numpy().tobytes()).hexdigest(),
                             "sft_episodes": len({row["episode"] for row in self.pool.rows
                                                 if row["split"] == "train" and row["success"]}),
                             "value_episodes": len({row["episode"] for row in self.pool.rows if row["split"] == "train"}),
                             "pool_audit": self.pool.audit, "shared_source_sha256": digest(ROOT/"src/vla_rl_mt/shared_value.py"),
                             "joint_source_sha256": digest(Path(__file__))}
            self.identity["training"] = {
                "steps": int(self.cfg.runner.max_steps), "seed": int(self.cfg.actor.seed),
                "global_batch_size": int(self.cfg.actor.global_batch_size),
                "micro_batch_size": int(self.cfg.actor.micro_batch_size),
                "value_weight": float(self.cfg.actor.model.value_weight),
                "adapter_rank": int(self.cfg.actor.model.adapter_rank),
                "head_width": int(self.cfg.actor.model.head_width), "readout": str(self.cfg.actor.model.readout),
                "learning_rate": float(self.cfg.actor.optim.lr), "value_lr": float(self.cfg.actor.optim.value_lr),
                "clip_grad": float(self.cfg.actor.optim.clip_grad), "weight_decay": float(self.cfg.actor.optim.weight_decay)}
            write_json(self.destination/"live-data.json", self.identity)
            self._first_step = True
            return ValueDataLoaderImpl(self.identity, loader), [], self.identity

        def _prepare_input(self, batch):
            raw = batch["observation"]
            observation = {key: to_observation(raw[key], self.device) for key in ("sft", "value")}
            observation.update({key: raw[key].to(self.device) for key in ("actions", "action_mask", "noise", "flow_time")})
            return observation, batch["target_values"].to(self.device), {}

        def set_global_step(self, step):
            self.global_step = int(step)
            if self._first_step:
                self.pairs.offset = int(step)*int(self.cfg.actor.global_batch_size)
                self.data_iter = iter(self.data_loader)
                self._first_step = False

        def optimizer_step(self):
            # Override just clipping, retaining the official AdamW/scheduler and
            # inherited training/backward loop. Single-GPU no-shard is enforced.
            self.optimizer_steps += 1
            self.grad_scaler.unscale_(self.optimizer)
            model = self.model.module
            inventory = gradient_inventory(model)
            if inventory["frozen_policy"]["gradient_tensors"] or not all(row["finite"] for row in inventory.values()):
                raise ValueError("Incorrect or nonfinite M2 gradient route")
            isolated_clip(model, float(self.cfg.actor.optim.clip_grad))
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            metrics = {key: sum(row[key] for row in model.step_metrics)/len(model.step_metrics)
                       for key in ("sft_loss", "value_loss")}
            model.step_metrics.clear()
            metrics.update(step=self.global_step+1, gradients=inventory,
                           peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
            with (self.destination/"joint-metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False)+"\n")
            return (inventory["adapter"]["gradient_l2"]**2+inventory["value_head"]["gradient_l2"]**2)**0.5, [
                group["lr"] for group in self.optimizer.param_groups]

        def run_eval(self):
            with self.worker_timer():
                return self._evaluate_values()

        def _evaluate_values(self):
            if self.cfg.runner.get("skip_value_eval", False):
                return {}
            model = self.model.module
            model.eval()
            rows = []
            with torch.no_grad():
                for index, row in enumerate(self.pool.rows):
                    if row["split"] != "val":
                        continue
                    raw, _, _ = self.pool.sample(index)
                    batch = torch.utils.data.default_collate([raw])
                    prediction = float(model.value(to_observation(batch, self.device), detach_shared=True)[0])
                    rows.append({**row, "prediction": prediction})
            task_metrics = {}
            for task in sorted({row["task"] for row in rows}):
                episodes = sorted({row["episode"] for row in rows if row["task"] == task})
                task_metrics[task] = {
                    name: float(np.mean([np.mean([abs(row["prediction"]-row["target"])**power
                        for row in rows if row["episode"] == episode]) for episode in episodes]))
                    for name, power in (("mae", 1), ("mse", 2))}
            metrics = {name: float(np.mean([row[name] for row in task_metrics.values()])) for name in ("mae", "mse")}
            write_json(self.destination/f"value-evaluation-step{self.global_step+1}.json",
                       {"rows": rows, "tasks": task_metrics, "macro_episode_then_task": metrics})
            return metrics

        def save_checkpoint(self, save_path, step=0):
            super().save_checkpoint(save_path, step)
            model = self.model.module
            torch.save({"kind": "M2 addon; pinned frozen policy required", "identity": self.identity,
                        "architecture": model.architecture, "step": step, "addon": model.addon_state()},
                       Path(save_path)/"addon.pt")
            write_json(Path(save_path)/"identity.json", self.identity)

        def load_checkpoint(self, load_path):
            identity = json.loads((Path(load_path)/"identity.json").read_text())
            if identity != self.identity:
                raise ValueError("Checkpoint data/model/sampling/source identity differs")
            super().load_checkpoint(load_path)

    return JointPrefixWorker
