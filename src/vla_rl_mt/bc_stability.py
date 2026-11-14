"""BC learning-curve exports; inherit the audited expert update unchanged."""

from pathlib import Path

import torch

from vla_rl_mt.expert_policy import make_worker_class as expert_worker
from vla_rl_mt.expert_policy import register, state_digest
from vla_rl_mt.joint_value import digest, write_json

__all__ = ["make_worker_class", "register"]


def make_worker_class():
    class BCTraceWorker(expert_worker()):
        def build_dataloader(self):
            result = super().build_dataloader()
            self.identity.update(trace_source_sha256=digest(Path(__file__)),
                evaluation_export_steps=list(self.cfg.runner.eval_export_steps),
                final_resumable=bool(self.cfg.runner.get("trace_final_resumable", True)))
            write_json(self.destination/"live-data.json", self.identity)
            return result

        def _mark_ready(self, path, step, resumable):
            marker = {"step": int(step), "checkpoint_sha256": digest(path/"addon.pt"),
                      "resumable": resumable, "kind": "full DCP" if resumable else "evaluation weights only"}
            temporary = path/"evaluation-ready.json.tmp"
            write_json(temporary, marker)
            temporary.replace(path/"evaluation-ready.json")

        def _export_evaluation(self, path, step):
            path.mkdir(parents=True, exist_ok=True)
            model = self.model.module
            expert = model.expert_state()
            temporary = path/"addon.pt.tmp"
            torch.save({"kind": "Evaluation-only expert/adapter; not a resumable optimizer checkpoint",
                "identity": self.identity, "architecture": model.architecture, "step": int(step),
                "addon": model.addon_state(), "expert": expert, "expert_sha256": state_digest(expert),
                "exposure": model.exposure}, temporary)
            temporary.replace(path/"addon.pt")
            write_json(path/"evaluation-only.json", {"resumable": False})
            write_json(path/"identity.json", self.identity)
            self._mark_ready(path, step, False)

        def optimizer_step(self):
            result = super().optimizer_step()
            step = self.global_step+1
            if step in self.cfg.runner.eval_export_steps and step != self.cfg.runner.max_steps:
                self._export_evaluation(self.destination/f"checkpoints/global_step_{step}/actor", step)
            return result

        def save_checkpoint(self, save_path, step=0):
            path = Path(save_path)
            if self.cfg.runner.get("trace_final_resumable", True):
                super().save_checkpoint(save_path, step)
                self._mark_ready(path, step, True)
            else:
                self._export_evaluation(path, step)

        def load_checkpoint(self, load_path):
            if (Path(load_path)/"evaluation-only.json").exists():
                raise ValueError("This point contains evaluation weights only; resume requires a full final DCP")
            super().load_checkpoint(load_path)

    return BCTraceWorker
