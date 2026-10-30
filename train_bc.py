"""Native RLinf SFT runner for task-balanced RoboTwin behavior cloning."""

import sys
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parent / "RLinf"),
]
from vla_mtrl.common import environment

environment()
import hydra
import torch.multiprocessing as mp
from rlinf.config import validate_cfg
from rlinf.runners.sft_runner import SFTRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from vla_mtrl import register
from vla_mtrl.worker import RoboTwinSFTWorker

register()


@hydra.main(version_base="1.1", config_path="configs", config_name="robotwin_bc")
def main(cfg):
    cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    actor = RoboTwinSFTWorker.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy("actor"),
    )
    runner = SFTRunner(cfg=cfg, actor=actor)
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
