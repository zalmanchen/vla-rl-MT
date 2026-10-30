"""RoboTwin research extensions; upstream RLinf stays unchanged."""

_registered = False


def register():
    global _registered
    if _registered:
        return
    from rlinf.models import register_model

    register_model("mtrl_frozen_pi0", build_model, category="embodied")
    _registered = True


def build_model(cfg, torch_dtype=None):
    from rlinf.models.embodiment.openpi import get_model

    model = get_model(cfg, torch_dtype)
    model.freeze_vlm()
    return model
