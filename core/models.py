import numpy as np

from core.sequences import one_hot


class ModelAdapter:
    """One interface for every model. If a gReLU API detail differs in your
    installed version, this file is the only place you need to touch."""

    def __init__(self, module, device, seq_len, task_names, meta, batch_size=None):
        self.module = module.to(device).eval()
        self.device = device
        self.seq_len = seq_len
        self.task_names = task_names
        self.meta = meta
        self.batch_size = batch_size or (16 if device != "cpu" else 4)

    def predict(self, seqs):
        import torch

        outs = []
        with torch.inference_mode():
            for i in range(0, len(seqs), self.batch_size):
                x = torch.from_numpy(one_hot(seqs[i:i + self.batch_size], self.seq_len)).to(self.device)
                y = self.module(x)
                if isinstance(y, (tuple, list)):
                    y = y[0]
                y = y.detach().float().cpu().numpy()
                if y.ndim == 2:
                    y = y[:, :, None]
                outs.append(y)
        return np.concatenate(outs, axis=0)


def _build_demo_model(seq_len, n_tasks, seed):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv1d(4, 32, 9, padding=4), nn.ReLU(),
        nn.Conv1d(32, 32, 9, padding=4), nn.ReLU(),
        nn.Conv1d(32, n_tasks, 1),
    )


def _load_lightning_ckpt(path):
    from grelu.lightning import LightningModel

    return LightningModel.load_from_checkpoint(path, map_location="cpu")


def _infer_seq_len(module):
    try:
        return int(module.data_params["train"]["seq_len"])
    except Exception:
        return None


def _task_names(module, n):
    try:
        t = module.data_params["tasks"]
        if isinstance(t, dict) and "name" in t:
            names = [str(x) for x in t["name"]]
        else:
            names = [str(x["name"]) if isinstance(x, dict) else str(x) for x in t]
        if len(names) == n:
            return names
    except Exception:
        pass
    return [f"task_{i}" for i in range(n)]


def load_model(spec: dict, device: str, batch_size=None):
    """spec examples:
      {"source": "demo", "seq_len": 200, "n_tasks": 3}
      {"source": "grelu_zoo", "project": "enformer", "model_name": "human"}
      {"source": "huggingface", "repo_id": "user/repo", "filename": "model.ckpt"}
      {"source": "checkpoint", "path": "C:/models/my.ckpt"}   # only trusted files (pickle!)
    """
    source = spec.get("source", "demo")
    warnings = []
    if source == "demo":
        n_tasks = int(spec.get("n_tasks", 3))
        module = _build_demo_model(int(spec.get("seq_len", 200)), n_tasks, int(spec.get("seed", 0)))
        warnings.append("Demo model has RANDOM weights: use it to test the pipeline only, not for biology.")
    elif source == "grelu_zoo":
        import grelu.resources

        module = grelu.resources.load_model(project=spec["project"], model_name=spec["model_name"])
    elif source == "huggingface":
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=spec["repo_id"], filename=spec["filename"], revision=spec.get("revision"))
        module = _load_lightning_ckpt(path)
        warnings.append("Checkpoints are pickle files: only load ones you trust.")
    elif source == "checkpoint":
        module = _load_lightning_ckpt(spec["path"])
        warnings.append("Checkpoints are pickle files: only load ones you trust.")
    else:
        raise ValueError(f"Unknown model source: {source}")

    seq_len = spec.get("seq_len") or _infer_seq_len(module)
    if not seq_len:
        raise ValueError("Could not infer model input length; pass model.seq_len in params.")
    n_tasks = int(spec.get("n_tasks", 3)) if source == "demo" else None
    if n_tasks is None:
        probe = ModelAdapter(module, device, int(seq_len), [], spec, batch_size)
        n_tasks = probe.predict(["N" * int(seq_len)]).shape[1]
    names = _task_names(module, n_tasks)
    adapter = ModelAdapter(module, device, int(seq_len), names, spec, batch_size)
    return adapter, warnings