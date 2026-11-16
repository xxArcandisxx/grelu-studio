"""Thin, tested helpers around gReLU's own API.
Every direct gReLU import lives here or in core/models.py, so a version change is a one-file fix."""
import numpy as np

from core.sequences import BASES


def grelu_version():
    try:
        from importlib.metadata import version
        return version("grelu")
    except Exception:  # noqa: BLE001
        return None


def require_grelu():
    try:
        import grelu  # noqa: F401
    except ImportError as e:
        raise ImportError("gReLU is not installed. Run: uv pip install grelu") from e


def to_nt(arr):
    """Normalize gReLU output (AnnData, (N,T) or (N,T,L)) to a plain (N, T) array."""
    if hasattr(arr, "X"):  # AnnData
        arr = arr.X
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim == 3:
        a = a.mean(axis=-1)
    if a.ndim != 2:
        raise ValueError(f"Unexpected gReLU output shape {a.shape}")
    return a


def build_transform(adapter, on_tasks, off_tasks=None, center_bins=0):
    """gReLU prediction transform that turns model output into one number per sequence.
    With off_tasks: on-target mean minus off-target mean (Specificity). Without: mean over tasks (Aggregate)."""
    require_grelu()
    from grelu.transforms.prediction_transforms import Aggregate, Specificity

    positions = None
    if center_bins and 0 < int(center_bins) < adapter.out_len:
        s = (adapter.out_len - int(center_bins)) // 2
        positions = list(range(s, s + int(center_bins)))
    if off_tasks:
        return Specificity(on_tasks=[int(i) for i in on_tasks], off_tasks=[int(i) for i in off_tasks],
                           on_aggfunc="mean", off_aggfunc="mean", off_weight=1.0, positions=positions,
                           length_aggfunc="mean", compare_func="subtract")
    return Aggregate(tasks=[int(i) for i in on_tasks], positions=positions,
                     length_aggfunc="mean", task_aggfunc="mean")


def ism_to_matrix(delta, ref_window):
    """gReLU ISM output -> attribution matrix.
    delta: DataFrame or array (4, w), rows A,C,G,T = (mutant - reference) score.
    Returns (4, w): the reference-base row holds -mean(delta over the 3 alternative bases),
    the same convention the built-in ISM uses."""
    d = delta.loc[list(BASES)].to_numpy() if hasattr(delta, "loc") else delta
    d = np.nan_to_num(np.asarray(d, dtype=float))
    if d.shape != (4, len(ref_window)):
        raise ValueError(f"ISM output shape {d.shape} does not match a window of {len(ref_window)} bases")
    mat = np.zeros_like(d)
    for j, base in enumerate(ref_window):
        if base in BASES:
            r = BASES.index(base)
            mat[r, j] = -d[[k for k in range(4) if k != r], j].mean()
    return mat


def attribution_to_matrix(attr, n_seqs):
    """gReLU get_attributions output -> (N, 4, L)."""
    a = np.asarray(attr, dtype=float)
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3 or a.shape[0] != n_seqs or a.shape[1] != 4:
        raise ValueError(f"Unexpected attribution shape {a.shape}")
    return a


def parse_evolve(df, n_seeds):
    """gReLU evolve() DataFrame -> one dict per starting sequence:
    final sequence, scores, and the history of accepted steps."""
    import pandas as pd

    out = []
    for i in range(n_seeds):
        sub = df[df["start_seq"] == i] if "start_seq" in df.columns else df
        best = sub[sub["best_in_iter"].astype(bool)].sort_values("iter")
        if best.empty:
            raise ValueError("evolve() returned no selected sequences")
        last = best.iloc[-1]
        hist = []
        for _, r in best.iterrows():
            pos = r.get("position")
            hist.append({"iter": int(r["iter"]), "score": float(r["total_score"]),
                         "position": None if pd.isna(pos) else int(pos),
                         "allele": None if pd.isna(r.get("allele")) else str(r.get("allele"))})
        out.append({"seq": str(last["seq"]), "score": float(last["total_score"]),
                    "prediction_score": float(last["prediction_score"]), "history": hist})
    return out


def diff_positions(a, b):
    """0-based positions where two equal-length sequences differ."""
    return [i for i, (x, y) in enumerate(zip(a, b)) if x != y]