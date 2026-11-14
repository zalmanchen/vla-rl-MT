"""Register the Motus 32-step shape without changing pinned RLinf sources.

The upstream RoboTwin TrainConfig uses a 50-step window. Its official SFT
loader reads that TrainConfig rather than the actor's YAML horizon. Register
one explicitly named clone so SFT, CFG and inference all use the same 32 steps.
No model, loss, optimizer or worker implementation is replaced.
"""

import dataclasses

CONFIG_NAME = "pi05_aloha_robotwin_h32"


def register():
    from rlinf.models.embodiment.openpi import dataconfig

    base = dataconfig.get_openpi_config("pi05_aloha_robotwin")
    config = dataclasses.replace(base, name=CONFIG_NAME,
                                 model=dataclasses.replace(base.model, action_horizon=32))
    existing = dataconfig._CONFIGS_DICT.get(CONFIG_NAME)
    if existing is not None and existing != config:
        raise ValueError(f"Conflicting OpenPI TrainConfig: {CONFIG_NAME}")
    dataconfig._CONFIGS_DICT[CONFIG_NAME] = config
