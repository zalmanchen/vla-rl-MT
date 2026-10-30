"""Use RLinf's native flow SFT worker with matched per-task exposure."""

import json
import math
from pathlib import Path

from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker
from torchdata.stateful_dataloader import StatefulDataLoader

from .data import BalancedSampler, RoboTwinWindows


class RoboTwinSFTWorker(FSDPVlaSftWorker):
    def build_dataloader(self, data_paths, eval_dataset=False):
        if eval_dataset or self._world_size != 1:
            raise ValueError(
                "Use one GPU per model; closed-loop evaluation is separate"
            )
        import openpi.transforms as transforms
        from rlinf.models.embodiment.openpi.transforms.pipeline import (
            build_openpi_transforms,
        )

        cfg = self.cfg
        inputs, _ = build_openpi_transforms(
            cfg.actor.model.model_path,
            "pi0_aloha_robotwin",
            {"norm_stats_path": cfg.actor.model.openpi_data.norm_stats_path},
        )
        dataset = RoboTwinWindows(
            data_paths, list(cfg.data.tasks), transforms.compose(inputs)
        )
        sampler = BalancedSampler(
            dataset.by_task, cfg.actor.micro_batch_size, cfg.actor.seed
        )
        workers = cfg.data.num_workers
        extra = (
            {
                "prefetch_factor": 2,
                "persistent_workers": True,
                "multiprocessing_context": "spawn",
            }
            if workers
            else {}
        )
        return StatefulDataLoader(
            dataset,
            batch_size=cfg.actor.micro_batch_size,
            sampler=sampler,
            num_workers=workers,
            pin_memory=True,
            **extra,
        ), None

    def get_train_model_output(self, batch):
        for task in batch["task_label"]:
            self._counts[task] = self._counts.get(task, 0) + 1
        return super().get_train_model_output(batch)

    def run_training(self):
        self._counts = {}
        metrics = super().run_training()
        for key in ["loss", "grad_norm"]:
            if not math.isfinite(float(metrics[key])):
                raise RuntimeError(f"Non-finite {key}")
        expected = self.global_batch_size // len(self.cfg.data.tasks)
        if set(self._counts) != set(self.cfg.data.tasks) or any(
            n != expected for n in self._counts.values()
        ):
            raise RuntimeError(f"Task exposure mismatch: {self._counts}")
        row = {
            "step": int(self.global_step) + 1,
            "task_samples": self._counts,
            **{
                key: float(metrics[key])
                for key in ["loss", "grad_norm", "learning_rate"]
            },
        }
        path = Path(self.cfg.runner.logger.log_path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "train_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        return metrics
