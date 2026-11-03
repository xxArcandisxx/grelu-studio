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
    arr = np.zeros((len(seqs), 4, L), dtype=np.float32)
    for i, s in enumerate(seqs):
        s, _ = fit_with_offset(s, L)
        idx = _LUT[np.frombuffer(s.encode(), dtype=np.uint8)]
        mask = idx >= 0
        arr[i, idx[mask], np.nonzero(mask)[0]] = 1.0
    return arr


def dinuc_shuffle(seq: str, rnd: random.Random, max_tries: int = 500) -> str:
    """Dinucleotide-preserving shuffle (random Eulerian walk, rejection sampling)."""
    n = len(seq)
    if n < 4:
        return seq
    graph = {}
    for a, b in zip(seq[:-1], seq[1:]):
        graph.setdefault(a, []).append(b)
    for _ in range(max_tries):
        edges = {v: rnd.sample(nb, len(nb)) for v, nb in graph.items()}
        ptr = {v: 0 for v in edges}
        cur = seq[0]
        out = [cur]
        while cur in edges and ptr[cur] < len(edges[cur]):
            nxt = edges[cur][ptr[cur]]
            ptr[cur] += 1
            out.append(nxt)
            cur = nxt
        if len(out) == n:
            return "".join(out)
    chars = list(seq)  # fallback: plain shuffle
    rnd.shuffle(chars)
    return "".join(chars)


def aggregate(preds: np.ndarray, center_bins) -> np.ndarray:
    """(N, T, L_out) -> (N, T), optionally using only the central bins."""
    L = preds.shape[-1]
    if center_bins and 0 < int(center_bins) < L:
        c = int(center_bins)
        s = (L - c) // 2
        preds = preds[..., s:s + c]
    return preds.mean(axis=-1)