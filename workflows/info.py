from core.device import detect_device
from core.models import load_model


def handle_info_workflow(params: dict) -> dict:
    import torch

    device = detect_device(params.get("device", "auto"))
    info = {"device": device, "torch": torch.__version__, "cuda": torch.cuda.is_available()}
    try:
        from importlib.metadata import version
        info["grelu"] = version("grelu")
    except Exception:
        info["grelu"] = "not installed"
    warnings = []
    if params.get("model"):
        adapter, warnings = load_model(params["model"], device)
        info.update({"seq_len": adapter.seq_len, "tasks": adapter.task_names})
    return {"device": device, "results": info, "evidence": [], "warnings": warnings}