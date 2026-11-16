import numpy as np


def rank(x):
    """Ranks starting at 0, ties get their average rank."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    return r if np.isfinite(r) else None


def spearman(a, b):
    return pearson(rank(a), rank(b))


def auroc(y_true, score):
    y = np.asarray(y_true).astype(int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = rank(score) + 1
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def expected_calibration_error(y, p, bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total, rows = 0.0, []
    for i in range(bins):
        upper = (p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1])
        m = (p >= edges[i]) & upper
        if m.sum() == 0:
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        total += float(m.mean()) * abs(conf - acc)
        rows.append({"bin": i, "n": int(m.sum()), "mean_pred": conf, "frac_pos": acc})
    return float(total), rows