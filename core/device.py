from core.utils import log


def detect_device(preferred: str = "auto") -> str:
    import torch

    has_cuda = torch.cuda.is_available()
    has_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    avail = {"cuda": has_cuda, "mps": has_mps, "cpu": True}
    if preferred != "auto":
        if avail.get(preferred):
            return preferred
        log(f"[engine] requested device '{preferred}' unavailable, auto-selecting")
    if has_cuda:
        return "cuda"
    if has_mps:
        return "mps"
    return "cpu"