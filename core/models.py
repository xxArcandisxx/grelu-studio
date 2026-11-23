import numpy as np
import operator

from core.sequences import fit_with_offset, one_hot
from core.utils import log, to_py
from core.model_lengths import length_contract, validate_length, numpy_metadata


def _norm3(y, n):
    """Normalize a prediction array to (N, T, L)."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError('Prediction batch count must be a positive integer.')
    with np.errstate(over='ignore', invalid='ignore'):
        y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y[:, :, None] if y.shape[0] == n else y[None]
    elif y.ndim == 1:
        y = y[:, None, None]
    from core.output_processing import prediction_tensor
    return prediction_tensor(y, n)


class ModelAdapter:
    """One interface for every model.

    backend "grelu"   -> predictions come from the model's own predict_on_seqs (gReLU)
    backend "builtin" -> predictions come from a plain PyTorch forward pass
    Gradient attributions and the randomized-weights copy always use plain PyTorch.
    """

    def __init__(self, module, device, seq_len, task_names, meta, batch_size=None, backend="builtin"):
        self.module = module.to(device).eval()
        self.device = device
        self.seq_len = seq_len
        self.task_names = task_names
        self.meta = meta
        if batch_size is None:
            batch_size = 16 if device != "cpu" else 4
        try:
            if isinstance(batch_size, bool) or operator.index(batch_size) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError('batch_size must be a positive integer.') from None
        self.batch_size = operator.index(batch_size)
        self.backend = backend
        self.is_grelu = hasattr(module, "predict_on_seqs") and hasattr(module, "data_params")
        self._use_grelu_predict = self.is_grelu and backend == "grelu"
        self._out_len = None
        self.notes = []  # runtime notes handlers can add to their warnings

    @property
    def grelu_device(self):
        """gReLU functions take 'cpu' or a GPU index. They do not support Apple MPS."""
        return 0 if self.device == "cuda" else "cpu"

    @property
    def out_len(self):
        if self._out_len is None:
            self._out_len = int(self._predict_torch(["N" * self.seq_len]).shape[-1])
        return self._out_len

    def predict(self, seqs):
        if self._use_grelu_predict:
            try:
                return self._predict_grelu(seqs)
            except Exception as e:  # noqa: BLE001
                msg = f"gReLU predict_on_seqs failed ({e}); fell back to the built-in forward pass."
                log("[engine] " + msg)
                self.notes.append(msg)
                self._use_grelu_predict = False
        return self._predict_torch(seqs)

    def _predict_grelu(self, seqs):
        outs = []
        for batch in self.iter_sequence_batches(seqs):
            chunk = [fit_with_offset(s, self.seq_len)[0] for s in batch]
            y = self.module.predict_on_seqs(chunk, device=self.grelu_device)
            outs.append(_norm3(y, len(chunk)))
        return np.concatenate(outs, axis=0)

    def _predict_torch(self, seqs):
        import torch

        outs = []
        with torch.inference_mode():
            for batch in self.iter_sequence_batches(seqs):
                x = torch.from_numpy(one_hot(batch, self.seq_len)).to(self.device)
                y = self.module(x)
                if isinstance(y, (tuple, list)):
                    y = y[0]
                outs.append(_norm3(y.detach().float().cpu().numpy(), x.shape[0]))
        return np.concatenate(outs, axis=0)

    def iter_sequence_batches(self, seqs):
        """Actual adapter chunking, independently iterable without model outputs."""
        for start in range(0, len(seqs), self.batch_size):
            yield seqs[start:start + self.batch_size]

    def grad_attribution(self, seqs, tasks, center_bins):
        """Gradient x input (with gradient correction). Returns (N, 4, seq_len);
        non-zero only at the reference base of each position."""
        import torch

        outs = []
        for batch in self.iter_sequence_batches(seqs):
            x = torch.from_numpy(one_hot(batch, self.seq_len)).to(self.device)
            x.requires_grad_(True)
            with torch.enable_grad():
                y = self.module(x)
                if isinstance(y, (tuple, list)):
                    y = y[0]
                if y.ndim == 2:
                    y = y[:, :, None]
                lo = y.shape[-1]
                if center_bins and 0 < int(center_bins) < lo:
                    st = (lo - int(center_bins)) // 2
                    y = y[..., st:st + int(center_bins)]
                score = y[:, list(tasks)].mean(dim=(1, 2)).sum()
                (grad,) = torch.autograd.grad(score, x)
            g = grad - grad.mean(dim=1, keepdim=True)
            outs.append((g * x).detach().float().cpu().numpy())
        return np.concatenate(outs, axis=0)

    def randomized_copy(self, seed=0):
        """A copy of this model with randomized weights (for the randomization sanity check)."""
        import copy

        import torch

        m = copy.deepcopy(self.module)
        gen = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            for prm in m.parameters():
                std = float(prm.std()) if prm.numel() > 1 else 1.0
                noise = torch.randn(prm.shape, generator=gen) * max(std, 1e-3)
                prm.copy_(noise.to(prm.device))
        result = ModelAdapter(m, self.device, self.seq_len, self.task_names, self.meta, self.batch_size, self.backend)
        if hasattr(self, 'length_contract'):
            result.length_contract = dict(self.length_contract)
        return result


def _build_demo_model(n_tasks, seed):
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

    import torch

    # Older gReLU checkpoints include NumPy arrays in their model metadata.
    with torch.serialization.safe_globals(numpy_metadata()):
        return LightningModel.load_from_checkpoint(path, map_location="cpu", weights_only=True)


def _task_names(module):
    """Task names from model.data_params['tasks'], or None if unavailable."""
    try:
        t = module.data_params["tasks"]
        if isinstance(t, dict) and "name" in t:
            return [str(x) for x in t["name"]]
        if isinstance(t, (list, tuple)):
            return [str(x["name"]) if isinstance(x, dict) else str(x) for x in t]
    except Exception:  # noqa: BLE001
        pass
    return None


def load_model(spec: dict, device: str, batch_size=None, backend: str = "auto"):
    """spec examples:
      {"source": "demo", "seq_len": 200, "n_tasks": 3}
      {"source": "grelu_zoo", "repo_id": "Genentech/human-atac-catlas-model", "filename": "model.ckpt"}
      {"source": "huggingface", "repo_id": "someone/their-grelu-model", "filename": "model.ckpt"}
      {"source": "checkpoint", "path": "C:/models/my.ckpt"}   # only trusted files (pickle!)
    backend: "auto" (gReLU functions for gReLU models, built-in otherwise) | "grelu" | "builtin"
    """
    if backend not in ("auto", "grelu", "builtin"):
        raise ValueError("backend must be 'auto', 'grelu' or 'builtin'")
    if spec.get("source") == "studio_trained":
        import torch
        payload = torch.load(spec["path"], map_location="cpu", weights_only=True)
        adapter, _ = load_model(payload["base_model"], device, batch_size, backend="builtin")
        validate_length(adapter.length_contract, spec.get('seq_len'))
        adapter.module.load_state_dict(payload["state_dict"])
        adapter.module.eval()
        adapter.meta = {**spec, "backend": "builtin"}
        adapter.task_names = payload["task_names"]
        return adapter, ["Locally trained checkpoint. Evaluate on independent data before drawing biological conclusions."]
    source = spec.get("source", "demo")
    warnings = []
    if source == "demo" and backend == "grelu":
        raise ValueError("backend 'grelu' needs a gReLU model (grelu_zoo, huggingface or checkpoint source). "
                         "The demo model supports 'builtin' only.")
    if source == "demo":
        module = _build_demo_model(int(spec.get("n_tasks", 3)), int(spec.get("seed", 0)))
        warnings.append("Demo model has RANDOM weights: use it to test the pipeline only, not for biology.")
    elif source in ("grelu_zoo", "huggingface"):
        if spec.get("project") or spec.get("model_name"):
            raise ValueError(
                "The gReLU model zoo moved from Weights & Biases to Hugging Face (gReLU >= 1.1.0). "
                "Use {'source': 'grelu_zoo', 'repo_id': 'Genentech/human-atac-catlas-model', 'filename': 'model.ckpt'}. "
                "Run the 'zoo' workflow with action 'list' to see every repo id.")
        if not spec.get("repo_id"):
            raise ValueError("model.repo_id is required, e.g. 'Genentech/human-atac-catlas-model'.")
        import grelu.resources

        module = grelu.resources.load_model(repo_id=spec["repo_id"], filename=spec.get("filename", "model.ckpt"),
                                            device="cpu")
        if not str(spec["repo_id"]).startswith("Genentech/"):
            warnings.append("This checkpoint is not from the official Genentech zoo. Checkpoints are pickle files: only load ones you trust.")
    elif source == "checkpoint":
        module = _load_lightning_ckpt(spec["path"])
        warnings.append("Checkpoints are pickle files: only load ones you trust.")
    else:
        raise ValueError(f"Unknown model source: {source}")

    is_grelu = hasattr(module, "predict_on_seqs") and hasattr(module, "data_params")
    if backend == "grelu" and not is_grelu:
        raise ValueError("backend 'grelu' needs a gReLU model (grelu_zoo, huggingface or checkpoint source). "
                         "The demo model supports 'builtin' only.")
    use = "grelu" if (backend == "grelu" or (backend == "auto" and is_grelu)) else "builtin"
    if use == "grelu" and device == "mps":
        device = "cpu"
        warnings.append("gReLU functions support only CPU and CUDA GPUs, so the Apple GPU (mps) is not used.")

    contract = validate_length(length_contract(getattr(module, 'data_params', None),
        getattr(module, 'model_params', None), demo_window=spec.get('seq_len', 200) if source == 'demo' else None),
        spec.get('seq_len'))
    seq_len = contract['seq_len']
    names = _task_names(module)
    adapter = ModelAdapter(module, device, int(seq_len), names or [], {**spec, "backend": use}, batch_size, use)
    adapter.length_contract = contract
    if not names:
        n_tasks = adapter._predict_torch(["N" * int(seq_len)]).shape[1]
        adapter.task_names = [f"task_{i}" for i in range(n_tasks)]
    return adapter, warnings


def model_card(adapter) -> dict:
    """Facts the UI shows on the model card (Step 2 of the pipeline)."""
    card = {"backend": adapter.backend, "device": adapter.device, "seq_len": adapter.seq_len,
            "n_tasks": len(adapter.task_names), "tasks": adapter.task_names[:500],
            "tasks_truncated": len(adapter.task_names) > 500}
    card['input_length_contract'] = getattr(adapter, 'length_contract', {
        'status': 'unknown', 'seq_len': None, 'verified_compatibility': False})
    dp = getattr(adapter.module, "data_params", None)
    if isinstance(dp, dict) and isinstance(dp.get("train"), dict):
        for k in ("genome", "bin_size", "label_len", "rc", "max_seq_shift", "n_seqs"):
            if k in dp["train"]:
                card[f"train_{k}"] = dp["train"][k]
    perf = getattr(adapter.module, "performance", None)
    try:
        if perf is not None and hasattr(perf, "to_dict"):
            perf = perf.to_dict()
        if perf:
            card["performance"] = to_py(perf)
    except Exception:  # noqa: BLE001
        pass
    return card
