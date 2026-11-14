"""Task gradient collection and matched actor optimizer baselines."""

import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from vla_rl_mt.joint_value import digest, write_json
from vla_rl_mt.signal_policy import build_model as signal_model, make_worker_class as signal_worker

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/"configs/multitask/policy_gradients.json"


def coefficients(a2, b2, dot, target2, mode):
    """Two-task balancing/PCGrad; match the ordinary raw norm when possible."""
    if mode not in ("mean", "balance", "pcgrad"):
        raise ValueError("Unknown gradient combination")
    ca, cb = 1., 1.
    nonzero = a2 > 1e-30 and b2 > 1e-30
    if mode == "balance" and nonzero:
        common = (math.sqrt(a2)+math.sqrt(b2))/2
        ca, cb = common/math.sqrt(a2), common/math.sqrt(b2)
    elif mode == "pcgrad" and nonzero and dot < 0:
        ca, cb = 1-dot/a2, 1-dot/b2
    output2 = max(0., ca*ca*a2+cb*cb*b2+2*ca*cb*dot)
    degenerate = output2 <= 1e-20*max(a2+b2, 1e-30)
    changed = (mode == "balance" and nonzero) or (mode == "pcgrad" and nonzero and dot < 0)
    if changed:
        if degenerate:
            ca, cb = 0., 0.
        else:
            factor = math.sqrt(max(target2, 0.)/output2)
            ca, cb = ca*factor, cb*factor
    cosine = dot/math.sqrt(a2*b2) if nonzero else None
    return ca, cb, {"a_norm": math.sqrt(a2), "b_norm": math.sqrt(b2),
        "cosine": max(-1., min(1., cosine)) if cosine is not None else None,
        "conflict": bool(nonzero and dot < 0), "zero_task": not nonzero,
        "intervention_applied": changed,
        "degenerate_sum": degenerate, "ordinary_norm": math.sqrt(max(target2, 0.)), "coefficient_a": ca, "coefficient_b": cb}


class SelectedPairs(Dataset):
    def __init__(self, pool, report, tasks, steps, seed):
        self.pool, self.offset = pool, 0
        lookup = {(r["episode"], r["observation_index"]): i for i, r in enumerate(pool.rows)}
        by_task = {task: [] for task in tasks}
        for i, chosen in enumerate(report["weights"]):
            if chosen:
                row = report["sequence"][i]
                task = pool.rows[lookup[row["episode"], row["observation_index"]]]["task"]
                by_task[task].append(i)
        if any(len(v) != 64 for v in by_task.values()):
            raise ValueError("Expected64 frozen selected draws per task")
        self.records, draws = [], []
        for step in range(steps):
            cycle, position = divmod(step, 32)
            for task_id, task in enumerate(tasks):
                for source in by_task[task][position*2:position*2+2]:
                    row = report["sequence"][source]
                    index = lookup[row["episode"], row["observation_index"]]
                    draws.append(index)
                    self.records.append({**row, "noise_key": row["noise_key"]+cycle*100000+seed*1000000,
                                         "source_draw": source, "cycle": cycle, "task_id": task_id})
        self.draws = torch.tensor(draws)
        self.weights = torch.ones(len(draws), dtype=torch.bool)

    def __len__(self): return len(self.draws)

    def __getitem__(self, index):
        index += self.offset
        row = int(self.draws[index])
        raw, actions, mask = self.pool.sample(row)
        generator = torch.Generator().manual_seed(self.records[index]["noise_key"])
        return {"native": raw, "actions": actions, "action_mask": mask,
            "noise": torch.randn(actions.shape, generator=generator),
            "flow_time": torch.rand((), generator=generator).pow(1/1.5)*.999+.001,
            "routing_random": torch.tensor(.5), "positive": torch.tensor(True),
            "success": torch.tensor(bool(self.pool.rows[row]["success"])),
            "task_id": torch.tensor(self.records[index]["task_id"])}


def build_model(config, torch_dtype=None):
    if digest(PROTOCOL) != config.gradient_protocol_sha256:
        raise ValueError("Gradient-study protocol changed")
    return signal_model(config, torch_dtype)


def register():
    from rlinf.models import register_model
    register_model("mt_policy_gradients", build_model, force=True)


def make_worker_class():
    from rlinf.data.datasets.recap.value_dataset import ValueDataLoaderImpl

    class PolicyGradientWorker(signal_worker()):
        def build_dataloader(self):
            super().build_dataloader()
            if self.cfg.actor.global_batch_size != 4 or self.cfg.actor.micro_batch_size != 2:
                raise ValueError("Gradient probe requires exactly two task-pure microbatches")
            settings = json.loads(PROTOCOL.read_text())
            if digest(PROTOCOL) != self.cfg.actor.model.gradient_protocol_sha256:
                raise ValueError("Sealed gradient protocol changed")
            path = Path(self.cfg.data.draw_weights_path)
            self.pairs = SelectedPairs(self.pool, json.loads(path.read_text()), settings["tasks"],
                                       int(self.cfg.runner.max_steps), int(self.cfg.actor.seed))
            self.gradient_mode = str(self.cfg.actor.model.gradient_mode)
            self._task_gradients = {}
            self.identity.update(scope=settings["scope"], gradient_mode=self.gradient_mode, actor_seed=int(self.cfg.actor.seed),
                gradient_protocol_sha256=digest(PROTOCOL), gradient_source_sha256=digest(Path(__file__)),
                sample_sequence_sha256=hashlib.sha256(json.dumps(self.pairs.records).encode()).hexdigest(),
                supervision_draws=len(self.pairs), selected_cycles=4, task_pure_microbatches=True)
            write_json(self.destination/"live-data.json", self.identity)
            write_json(self.destination/"selected-sequence.json", self.pairs.records)
            loader = DataLoader(self.pairs, batch_size=2, num_workers=0)
            return ValueDataLoaderImpl(self.identity, loader), [], self.identity

        def _prepare_input(self, batch):
            data = super()._prepare_input(batch)
            task = batch["task_id"]
            if not bool((task == task[0]).all()):
                raise ValueError("Microbatch contains multiple tasks")
            self._micro_task = int(task[0])
            return data

        def _groups(self):
            model = self.model.module
            return {"adapter": list(model.adapter.parameters()), "expert": list(model.expert.parameters())}

        def before_micro_batch(self, model, is_last_micro_batch):
            original = super().before_micro_batch(model, is_last_micro_batch)
            if self.gradient_mode == "native_replay": return original

            @contextmanager
            def capture():
                with original: yield
                task = self._micro_task
                if task != int(is_last_micro_batch) or task in self._task_gradients:
                    raise ValueError("Unexpected task ordering or duplicate backward capture")
                scale = float(self.grad_scaler.get_scale())
                snapshot = {}
                for name, parameters in self._groups().items():
                    snapshot[name] = (torch.cat([(p.grad.detach().float().reshape(-1) if p.grad is not None
                        else torch.zeros(p.numel(), device=p.device)) for p in parameters])/scale,
                        [p.grad is not None for p in parameters])
                self._task_gradients[task] = snapshot
                self.model.zero_grad(set_to_none=True)
            return capture()

        def optimizer_step(self):
            if self.gradient_mode == "native_replay": return super().optimizer_step()
            if set(self._task_gradients) != {0, 1}:
                raise ValueError("Missing a task gradient")
            a, b = self._task_gradients[0], self._task_gradients[1]
            a2 = sum(float(torch.dot(v[0], v[0])) for v in a.values())
            b2 = sum(float(torch.dot(v[0], v[0])) for v in b.values())
            dot = sum(float(torch.dot(a[k][0], b[k][0])) for k in a)
            ordinary2 = sum(float((a[k][0]+b[k][0]).square().sum()) for k in a)
            ca, cb, metrics = coefficients(a2, b2, dot, ordinary2, self.gradient_mode)
            groups = self._groups()
            before, group_metrics = {}, {}
            scale = float(self.grad_scaler.get_scale())
            for name, parameters in groups.items():
                av, amask = a[name]; bv, bmask = b[name]
                combined = ca*av+cb*bv
                if not bool(torch.isfinite(combined).all()): raise ValueError("Nonfinite combined gradient")
                group_metrics[name] = {"a_norm": float(av.norm()), "b_norm": float(bv.norm()),
                    "dot": float(torch.dot(av, bv)), "combined_preclip_norm": float(combined.norm())}
                offset = 0
                for i, p in enumerate(parameters):
                    count = p.numel()
                    p.grad = (combined[offset:offset+count].view_as(p)*scale).to(p.dtype) if amask[i] or bmask[i] else None
                    offset += count
                before[name] = torch.cat([p.detach().float().reshape(-1) for p in parameters])
                del combined
            result = super().optimizer_step()
            for name, parameters in groups.items():
                delta = torch.cat([p.detach().float().reshape(-1) for p in parameters])-before[name]
                group_metrics[name].update(actual_delta_norm=float(delta.norm()),
                    a_dot_actual_delta=float(torch.dot(a[name][0], delta)), b_dot_actual_delta=float(torch.dot(b[name][0], delta)))
            metrics.update(step=self.global_step+1, mode=self.gradient_mode, groups=group_metrics, grad_scale=scale)
            with (self.destination/"task-gradients.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False)+"\n")
            self._task_gradients.clear()
            return result

    return PolicyGradientWorker
