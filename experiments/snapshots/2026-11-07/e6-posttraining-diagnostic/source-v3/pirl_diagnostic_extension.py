"""Select a source-defined diagnostic worker that Ray can serialize reliably."""
_registered = False


def register():
    global _registered
    if _registered:
        return
    _registered = True
    import pirl_extension
    pirl_extension.register()
    import rlinf.workers.actor.embodied_fsdp_actor_worker as actor_module
    from pirl_diagnostic_actor import DiagnosticEmbodiedFSDPActor
    actor_module.EmbodiedFSDPActor = DiagnosticEmbodiedFSDPActor
