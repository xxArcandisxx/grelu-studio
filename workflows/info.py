from core.device import detect_device
from core.grelu_api import grelu_version
from core.models import load_model, model_card


def handle_info_workflow(params: dict) -> dict:
    """params: device, and optionally model (+ backend) to get a model card."""
    import torch

    device = detect_device(params.get("device", "auto"))
    has_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    info = {"device": device, "torch": torch.__version__, "cuda": torch.cuda.is_available(), "mps": has_mps,
            "grelu": grelu_version() or "not installed"}
    warnings = []
    if params.get("model"):
        adapter, warnings = load_model(params["model"], device, backend=params.get("backend", "auto"))
        info["model"] = model_card(adapter)
        device = adapter.device
    return {"device": device, "results": info, "evidence": [], "warnings": warnings}