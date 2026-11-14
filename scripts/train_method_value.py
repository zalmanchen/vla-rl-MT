"""Run an M1 data/model extension through RLinf's unmodified value training loop."""

import importlib
import os

import hydra
import torch.multiprocessing as mp

extension = importlib.import_module(os.environ.get("RLINF_EXT_MODULE", "vla_rl_mt.value_head"))
make_worker_class, register = extension.make_worker_class, extension.register

register()
mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1", config_path=None, config_name=None)
def main(config):
    from rlinf.config import validate_cfg
    from rlinf.runners.sft_runner import SFTRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement

    namespace = config.runner.get("ray_namespace", None)
    if namespace:
        Cluster.NAMESPACE = str(namespace)
    config = validate_cfg(config)
    cluster = Cluster(cluster_cfg=config.cluster)
    placement = HybridComponentPlacement(config, cluster).get_strategy("actor")
    actor = make_worker_class().create_group(config).launch(
        cluster, name=config.actor.group_name, placement_strategy=placement)
    runner = SFTRunner(cfg=config, actor=actor)
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
