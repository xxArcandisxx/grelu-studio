import random

import numpy as np

BASES = "ACGT"
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

_LUT = np.full(256, -1, dtype=np.int16)
for _i, _b in enumerate(BASES):
    _LUT[ord(_b)] = _i
    _LUT[ord(_b.lower())] = _i


def revcomp(s: str) -> str:
    return s.translate(COMP)[::-1]


def fit_with_offset(seq: str, L: int):
    """Center-crop or N-pad to length L. Returns (seq, shift); new_pos = old_pos + shift."""
    n = len(seq)
    if n == L:
        return seq, 0
    if n > L:
        start = (n - L) // 2
        return seq[start:start + L], -start
    pad = L - n
    left = pad // 2
    return "N" * left + seq + "N" * (pad - left), left


def one_hot(seqs, L: int) -> np.ndarray:
    # Gate 2 normalizes website input; the encoder must also fail clearly if a
    # caller bypasses that boundary. Never turn invalid characters into N.
    if isinstance(L, bool) or not isinstance(L, (int, np.integer)) or L <= 0:
        raise ValueError('Encoding window must be a positive integer.')
    if isinstance(seqs, str) or not isinstance(seqs, (list, tuple)) or not seqs:
        raise ValueError('Encoding requires a nonempty sequence batch.')
    for s in seqs:
        if not isinstance(s, str) or not s or not s.isascii() or any(base not in 'ACGTNacgtn' for base in s):
            raise ValueError('Encoding requires nonempty ASCII A/C/G/T/N sequences; validate and normalize input first.')
    arr = np.zeros((len(seqs), 4, L), dtype=np.float32)
    for i, s in enumerate(seqs):
        s, _ = fit_with_offset(s, L)
        idx = _LUT[np.frombuffer(s.encode(), dtype=np.uint8)]
        mask = idx >= 0
        arr[i, idx[mask], np.nonzero(mask)[0]] = 1.0
    return arr


def dinuc_shuffle(seq: str, rnd: random.Random, max_tries: int = 500) -> str:
    """Shuffle directed dinucleotide edges into a randomized Eulerian trail.

    Hierholzer's algorithm consumes every edge, preserving all dinucleotide
    counts even for repetitive sequences. max_tries remains for API compatibility.
    The samples are randomized trails, not a claim of uniform sampling.
    """
    if len(seq) < 3:
        return seq
    edges = {}
    for left, right in zip(seq[:-1], seq[1:]):
        edges.setdefault(left, []).append(right)
    for neighbors in edges.values():
        rnd.shuffle(neighbors)
    stack, trail = [seq[0]], []
    while stack:
        neighbors = edges.get(stack[-1])
        if neighbors:
            stack.append(neighbors.pop())
        else:
            trail.append(stack.pop())
    return "".join(reversed(trail))


def aggregate(preds: np.ndarray, center_bins) -> np.ndarray:
    """(N, T, L_out) -> (N, T), optionally using only the central bins."""
    from core.output_processing import prediction_tensor
    preds = prediction_tensor(preds)
    L = preds.shape[-1]
    if center_bins and 0 < int(center_bins) < L:
        start = (L - int(center_bins)) // 2
        preds = preds[..., start:start + int(center_bins)]
    if preds.shape[-1] == 1:
        return preds[..., 0].astype(np.float64)
    # Accumulate in float64 so a finite float32 mean cannot overflow its sum.
    try:
        with np.errstate(over='raise', invalid='raise'):
            return preds.mean(axis=-1, dtype=np.float64)
    except FloatingPointError:
        values = preds.astype(np.float64)
        scale = np.max(np.abs(values), axis=-1, keepdims=True)
        scaled = np.divide(values, scale, out=np.zeros_like(values), where=scale != 0)
        return scaled.mean(axis=-1) * scale[..., 0]
