"""M4 update-scope control: native action expert plus the same prefix adapter.

Register only expert modules with FSDP; rebind native call sites to their wrapped
views so mixed original dtypes and actual FSDP forward hooks are preserved.
Frozen VLM weights stay outside checkpoints. No native/CFG parity is implied.
"""

import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn

from vla_rl_mt.joint_value import digest, write_json
from vla_rl_mt.policy_update import NativePolicyUpdate

ROOT = Path(__file__).resolve().parents[2]


def state_digest(state):
    result = hashlib.sha256()
    for key, value in sorted(state.items()):
        result.update(key.encode())
        result.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def expert_bindings(policy):
    bindings = []
    for index, block in enumerate(policy.llm.layers):
        for name, parent in [("attention_norm", block.pre_attention_norms),
                             ("ffw_norm", block.pre_ffw_norms), ("mlp", block.mlps),
                             *[(name, getattr(block.attn, name)) for name in ("q_proj", "k_proj", "v_proj", "o_proj")]]:
            if parent[1] is not None:
                bindings.append((f"block{index:02d}_{name}", parent, "1"))
    bindings.append(("final_norm", policy.llm.final_norms, "1"))
    for name in ("action_in_proj", "action_out_proj", "state_proj", "action_time_mlp_in",
                 "action_time_mlp_out", "time_mlp_in", "time_mlp_out"):
        if getattr(policy, name, None) is not None:
            bindings.append((name, policy, name))
    return bindings


def inventory(model):
    expert_ids = {id(param) for param in model.expert.parameters()}
    groups = {"frozen_vlm": [param for param in model.policy.parameters() if id(param) not in expert_ids],
              "action_expert": list(model.expert.parameters()), "adapter": list(model.adapter.parameters()),
              "value_head": list(model.value_head.parameters())}
    result = {}
    for name, parameters in groups.items():
        gradients = [param.grad for param in parameters if param.grad is not None]
        result[name] = {"parameters": sum(param.numel() for param in parameters),
            "trainable": sum(param.numel() for param in parameters if param.requires_grad),
            "gradient_tensors": len(gradients),
            "gradient_l2": sum(float(grad.float().square().sum()) for grad in gradients)**0.5,
            "finite": all(bool(torch.isfinite(grad).all()) for grad in gradients)}
    return result


def clip_actor(model, max_norm=1.0):
    # Preserve the old adapter clipping rule; the added expert is clipped
    # separately so its norm does not indirectly rescale adapter gradients.
    for module in (model.adapter, model.expert):
        nn.utils.clip_grad_norm_(module.parameters(), max_norm, error_if_nonfinite=True)


class ExpertPolicyUpdate(NativePolicyUpdate):
    def __init__(self, policy, mode, *, adapter_lr, expert_lr, update_scope="expert_adapter", **kwargs):
        super().__init__(policy, mode, **kwargs)
        if update_scope not in ("adapter_only", "expert_adapter"):
            raise ValueError("Unknown actor update scope")
        self.update_scope = update_scope
        policy.requires_grad_(True)
        policy.freeze_vlm(freeze_action_expert=False)
        self.bindings = expert_bindings(policy)
        self.expert = nn.ModuleDict({name: getattr(parent, attribute) for name, parent, attribute in self.bindings})
        for name, module in self.expert.items():
            module._fsdp_wrap_name = name
        expected = {id(param) for param in policy.parameters() if param.requires_grad}
        if expected != {id(param) for param in self.expert.parameters()}:
            raise ValueError("Registered expert modules do not cover exactly the trainable native policy")
        # Keep identical forward/autograd paths in both controls. Toggling
        # requires_grad alone changes native BF16 numerics. The adapter-only
        # control computes expert gradients, excludes them from the optimizer,
        # then clears them. Expert weights are certified unchanged by digest.
        self.lr_multipliers = {"expert.": expert_lr/adapter_lr}

    @property
    def _no_split_names(self):
        return list(self.expert)

    def bind_expert(self):
        for name, parent, attribute in self.bindings:
            setattr(parent, attribute, self.expert[name])

    def expert_state(self):
        return {key: value.detach().cpu().clone() for key, value in self.expert.state_dict().items()}


def build_model(config, torch_dtype=None):
    sys.path.insert(0, str(ROOT/"scripts"))
    from evaluate_multitask import load_model

    plan = json.loads((ROOT/"configs/multitask/method_m4.json").read_text())
    manifest = json.loads((Path(config.pool_path)/"manifest.json").read_text())
    parent_path = ROOT/plan["w1_checkpoint"]
    if (digest(parent_path) != plan["parent_checkpoint_sha256"]["w1"]
            or digest(ROOT/manifest["protocol"]["model"]["path"]/"model.safetensors")
                != manifest["protocol"]["model"]["sha256"]):
        raise ValueError("Pinned parent/base changed")
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    policy, _ = load_model("native", manifest["protocol"])
    policy.gradient_checkpointing_enable({"use_reentrant": False})
    model = ExpertPolicyUpdate(policy, str(config.mode), adapter_lr=float(config.adapter_lr),
                              expert_lr=float(config.expert_lr), update_scope=str(config.update_scope),
                              **parent["architecture"]).to("cuda").train()
    model.load_addon(parent["addon"])
    return model


def register():
    from rlinf.models import register_model

    register_model("mt_native_expert_policy", build_model, force=True)


def make_worker_class():
    from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager

    from vla_rl_mt.policy_update import make_worker_class as policy_worker

    class ExpertPolicyWorker(policy_worker()):
        def build_optimizer(self, model, enable_critic_warmup=False):
            expert = model.module.expert
            control = model.module.update_scope == "adapter_only"
            if control:
                expert.requires_grad_(False)
            try:
                return super().build_optimizer(model, enable_critic_warmup)
            finally:
                expert.requires_grad_(True)

        def build_dataloader(self):
            result = super().build_dataloader()
            self.identity.update(scope="native action-expert plus adapter scope control; VLM and labels frozen",
                update_scope=str(self.cfg.actor.model.update_scope),
                expert_source_sha256=digest(Path(__file__)), expert_lr=float(self.cfg.actor.model.expert_lr),
                clipping="adapter and expert separately, same max_norm", original_parameter_dtypes_preserved=True)
            write_json(self.destination/"live-data.json", self.identity)
            return result

        def setup_model_and_optimizer(self):
            super().setup_model_and_optimizer()
            model = self.model.module
            model.bind_expert()
            expected = {id(param) for param in model.adapter.parameters()}
            if model.update_scope == "expert_adapter":
                expected.update(id(param) for param in model.expert.parameters())
            actual = {id(param) for group in self.optimizer.param_groups for param in group["params"]}
            if expected != actual:
                raise ValueError("Optimizer does not own exactly the trainable actor parameters")
            # Initialize state-dict traversal from the real FSDP root before
            # querying any child wrappers through native alias call sites.
            root_state = self.model.state_dict()
            expert_state = {key.removeprefix("expert."): value for key, value in root_state.items()
                            if key.startswith("expert.")}
            write_json(self.destination/"parameter-audit.json", {"groups": inventory(model),
                "initial_expert_sha256": state_digest(expert_state),
                "optimizer_groups": [{"lr": group["lr"], "parameters": sum(p.numel() for p in group["params"])}
                                     for group in self.optimizer.param_groups],
                "optimizer": type(self.optimizer).__name__, "optimizer_coverage_exact": True,
                "expert_gradients_computed_in_both_scopes": True,
                "fsdp_expert_call_sites_rebound": True})

        def optimizer_step(self):
            self.optimizer_steps += 1
            self.grad_scaler.unscale_(self.optimizer)
            model = self.model.module
            groups = inventory(model)
            if (groups["frozen_vlm"]["trainable"] or groups["frozen_vlm"]["gradient_tensors"]
                    or groups["value_head"]["gradient_tensors"] or not all(row["finite"] for row in groups.values())):
                raise ValueError("Incorrect or nonfinite expert-scope gradient route")
            clip_actor(model, float(self.cfg.actor.optim.clip_grad))
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.model.zero_grad(set_to_none=True)
            metrics = {key: sum(row[key] for row in model.step_metrics)/len(model.step_metrics)
                       for key in ("sft_loss", "value_loss")}
            model.step_metrics.clear()
            metrics.update(step=self.global_step+1, gradients=groups,
                           peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
            with (self.destination/"joint-metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False)+"\n")
            return (groups["adapter"]["gradient_l2"]**2+groups["action_expert"]["gradient_l2"]**2)**0.5, [
                group["lr"] for group in self.optimizer.param_groups]

        def save_checkpoint(self, save_path, step=0):
            FSDPModelManager.save_checkpoint(self, save_path, step)
            model = self.model.module
            expert = model.expert_state()
            torch.save({"kind": "M4 action expert plus adapter; pinned frozen VLM required", "identity": self.identity,
                        "architecture": model.architecture, "step": step, "addon": model.addon_state(),
                        "expert": expert, "expert_sha256": state_digest(expert),
                        "exposure": model.exposure}, Path(save_path)/"addon.pt")
            write_json(Path(save_path)/"identity.json", self.identity)
            write_json(self.destination/"exposure.json", model.exposure)

        def load_checkpoint(self, load_path):
            if json.loads((Path(load_path)/"identity.json").read_text()) != self.identity:
                raise ValueError("Checkpoint data/model/sampling/source identity differs")
            FSDPModelManager.load_checkpoint(self, load_path)
            checkpoint = torch.load(Path(load_path)/"addon.pt", map_location="cpu", weights_only=True)
            self.model.module.exposure.update(checkpoint["exposure"])

    return ExpertPolicyWorker
