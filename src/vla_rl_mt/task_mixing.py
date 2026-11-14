"""Original-Motus BC with matched per-task example and flow-noise streams."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from vla_rl_mt.bc_stability import make_worker_class as trace_worker
from vla_rl_mt.expert_policy import ExpertPolicyUpdate, state_digest
from vla_rl_mt.joint_value import digest, draw_indices, write_json
from vla_rl_mt.policy_update import GuidedPool

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/"configs/multitask/task_mixing.json"


class TaskPairs(Dataset):
    """A mixed batch contains two examples from each independent task stream."""

    def __init__(self, pool, group, steps, *, tasks, seed=0):
        if group not in ("a_only", "b_only", "mixed") or len(tasks) != 2 or not 0 < steps <= 128:
            raise ValueError("Expected two tasks, a declared group and1..128 steps")
        self.pool, self.offset = pool, 0
        self.draws, self.noise_keys, self.task_indices = [], [], []
        streams = []
        for task_index, task in enumerate(tasks):
            candidates = [i for i, row in enumerate(pool.rows) if row["split"] == "train" and row["success"] and row["task"] == task]
            if not candidates:
                raise ValueError(f"No successful training episodes for {task}")
            streams.append(draw_indices(pool.rows, candidates, 512, seed+11700+task_index))
        for step in range(steps):
            selected = (0, 1) if group == "mixed" else ((0,) if group == "a_only" else (1,))
            per_task = 2 if group == "mixed" else 4
            for task_index in selected:
                for occurrence in range(step*per_task, (step+1)*per_task):
                    self.draws.append(int(streams[task_index][occurrence]))
                    self.task_indices.append(task_index)
                    self.noise_keys.append(seed+3_000_000+task_index*10000+occurrence)
        self.draws = torch.tensor(self.draws)

    def __len__(self):
        return len(self.draws)

    def __getitem__(self, index):
        index += self.offset
        raw, actions, mask = self.pool.sample(int(self.draws[index]))
        generator = torch.Generator().manual_seed(self.noise_keys[index])
        return {"native": raw, "actions": actions, "action_mask": mask,
                "noise": torch.randn(actions.shape, generator=generator),
                "flow_time": torch.rand((), generator=generator).pow(1/1.5)*.999+.001,
                "routing_random": torch.rand((), generator=generator),
                "positive": torch.tensor(True), "success": torch.tensor(True)}

    def sequence(self):
        return [{"episode": self.pool.rows[int(index)]["episode"],
                 "observation_index": self.pool.rows[int(index)]["observation_index"],
                 "task": self.pool.rows[int(index)]["task"], "noise_key": noise}
                for index, noise in zip(self.draws, self.noise_keys, strict=True)]


def native_policy(protocol):
    sys.path.insert(0, str(ROOT/"scripts"))
    from evaluate_multitask import load_model

    if digest(ROOT/protocol["model"]["path"]/"model.safetensors") != protocol["model"]["sha256"]:
        raise ValueError("Pinned original Motus weights changed")
    policy, _ = load_model("native", protocol)
    policy.gradient_checkpointing_enable({"use_reentrant": False})
    return policy


def attach(policy, settings):
    with torch.random.fork_rng():
        torch.manual_seed(settings["seed"])
        model = ExpertPolicyUpdate(policy, "bc", adapter_lr=settings["adapter_lr"], expert_lr=settings["expert_lr"],
                                   update_scope=settings["update_scope"], **settings["architecture"])
    return model.to("cuda")


def build_model(config, torch_dtype=None):
    if digest(PROTOCOL) != config.protocol_sha256:
        raise ValueError("Task-mixing protocol changed after seal")
    settings = json.loads(PROTOCOL.read_text())
    path = Path(config.initial_addon_path)
    if digest(path) != config.initial_addon_sha256:
        raise ValueError("Common original-Motus addon changed")
    initial = torch.load(path, map_location="cpu", weights_only=True)
    manifest = json.loads((Path(config.pool_path)/"manifest.json").read_text())
    if initial["base_sha256"] != manifest["protocol"]["model"]["sha256"]:
        raise ValueError("Addon belongs to a different base")
    model = attach(native_policy(manifest["protocol"]), settings).train()
    model.load_addon(initial["addon"])
    if state_digest(model.expert_state()) != initial["expert_sha256"]:
        raise ValueError("Native expert initialization differs from common base")
    return model


def register():
    from rlinf.models import register_model

    register_model("mt_native_task_policy", build_model, force=True)


def make_worker_class():
    from rlinf.data.datasets.recap.value_dataset import ValueDataLoaderImpl

    class TaskMixingWorker(trace_worker()):
        def build_dataloader(self):
            if (self._world_size != 1 or not self.cfg.actor.fsdp_config.use_orig_params
                    or self.cfg.actor.fsdp_config.sharding_strategy != "no_shard"):
                raise ValueError("Task controls require single-GPU NO_SHARD/original parameters")
            settings = json.loads(PROTOCOL.read_text())
            if digest(PROTOCOL) != self.cfg.actor.model.protocol_sha256:
                raise ValueError("Sealed task-control protocol changed")
            self.pool = GuidedPool(self.cfg.data.pool_path, self.cfg.data.pool_manifest_sha256)
            self.pairs = TaskPairs(self.pool, str(self.cfg.actor.model.task_group), int(self.cfg.runner.max_steps),
                                   tasks=settings["tasks"], seed=int(self.cfg.actor.seed))
            loader = DataLoader(self.pairs, batch_size=int(self.cfg.actor.micro_batch_size), num_workers=0)
            self.destination = Path(self.cfg.runner.logger.log_path)/self.cfg.runner.logger.experiment_name
            self.destination.mkdir(parents=True, exist_ok=True)
            sequence = self.pairs.sequence()
            self.identity = {"scope": settings["scope"], "mode": "bc", "task_group": str(self.cfg.actor.model.task_group),
                "pool_manifest_sha256": str(self.cfg.data.pool_manifest_sha256),
                "parent_sha256": str(self.cfg.actor.model.initial_addon_sha256),
                "base_sha256": self.pool.manifest["protocol"]["model"]["sha256"],
                "protocol_sha256": digest(PROTOCOL), "source_sha256": digest(Path(__file__)),
                "sample_sequence_sha256": hashlib.sha256(json.dumps(sequence).encode()).hexdigest(),
                "task_exposure": dict(Counter(row["task"] for row in sequence)), "label_sha256": None,
                "steps": int(self.cfg.runner.max_steps), "global_batch": int(self.cfg.actor.global_batch_size),
                "micro_batch": int(self.cfg.actor.micro_batch_size), "lr": float(self.cfg.actor.optim.lr),
                "expert_lr": float(self.cfg.actor.model.expert_lr), "update_scope": "expert_adapter",
                "deploy_prompt": "plain", "clipping": "adapter/expert separately at1",
                "evaluation_export_steps": list(self.cfg.runner.eval_export_steps),
                "final_resumable": bool(self.cfg.runner.get("trace_final_resumable", True))}
            write_json(self.destination/"live-data.json", self.identity)
            write_json(self.destination/"sample-sequence.json", sequence)
            self._first_step = True
            return ValueDataLoaderImpl(self.identity, loader), [], self.identity

    return TaskMixingWorker
