"""Select a source-defined diagnostic worker that Ray can serialize reliably."""
_registered = False


def register():
    global _registered
    if _registered:
        return
    _registered = True
    import pirl_extension
    pirl_extension.register()
    import torch
    import rlinf.models.embodiment.openpi as factory
    build = factory.get_model

    def build_with_precision(cfg, torch_dtype=None):
        model = build(cfg, torch_dtype)
        if cfg.get('pilot_strict_fp32', False):
            # Pi0.__init__ sets "high" globally, so apply this after construction.
            torch.set_float32_matmul_precision('highest')
            torch.backends.cudnn.allow_tf32 = False
            print('[E6 precision] matmul=highest; cudnn_tf32=False', flush=True)
        return model

    factory.get_model = build_with_precision
    import rlinf.workers.actor.embodied_fsdp_actor_worker as actor_module
    from pirl_diagnostic_actor import DiagnosticEmbodiedFSDPActor
    actor_module.EmbodiedFSDPActor = DiagnosticEmbodiedFSDPActor
