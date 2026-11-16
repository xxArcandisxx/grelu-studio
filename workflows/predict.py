import csv
import random
from pathlib import Path

import numpy as np

from core.device import detect_device
from core.io import read_sequences, sequence_qc
from core.models import load_model
from core.sequences import aggregate, dinuc_shuffle, fit_with_offset, revcomp
from core.stats import pearson
from core.utils import card, log


def _downsample(track, n_bins):
    if track.shape[-1] <= n_bins:
        return track
    parts = np.array_split(track, n_bins, axis=-1)
    return np.stack([p.mean(axis=-1) for p in parts], axis=-1)


def handle_predict_workflow(params: dict) -> dict:
    """
    params:
      model, device, ONE of (sequences | fasta | sequences_csv), center_bins(0=all),
      n_shuffles(5), return_tracks(true), track_bins(100), max_sequences(1000), seed, output_dir
    """
    device = detect_device(params.get("device", "auto"))
    adapter, warnings = load_model(params["model"], device, params.get("batch_size"))
    seqs, w = read_sequences(params)
    warnings += w
    L, names = adapter.seq_len, adapter.task_names
    T, N = len(names), len(seqs)
    qc, w = sequence_qc(seqs, L)
    warnings += w
    center = params.get("center_bins", 0)
    n_sh = int(params.get("n_shuffles", 5))
    track_bins = int(params.get("track_bins", 100))
    rnd = random.Random(int(params.get("seed", 0)))

    fitted = [fit_with_offset(s, L)[0] for _, s in seqs]
    log(f"[engine] predict: {N} sequences on {device}")
    preds = adapter.predict(fitted)  # (N, T, L_out)
    scores = aggregate(preds, center)
    rc_scores = aggregate(adapter.predict([revcomp(s) for s in fitted]), center)

    base = None
    if n_sh > 0:
        base = np.zeros((N, T))
        for i, s in enumerate(fitted):
            sh = [dinuc_shuffle(s, rnd) for _ in range(n_sh)]
            base[i] = aggregate(adapter.predict(sh), center).mean(axis=0)

    rows = []
    for i, (sid, s) in enumerate(seqs):
        acgt = sum(s.count(b) for b in "ACGT")
        row = {"id": sid, "length": len(s),
               "gc_content": (s.count("G") + s.count("C")) / max(acgt, 1),
               "scores": {names[t]: float(scores[i, t]) for t in range(T)},
               "reverse_complement_scores": {names[t]: float(rc_scores[i, t]) for t in range(T)}}
        if base is not None:
            row["shuffled_baseline"] = {names[t]: float(base[i, t]) for t in range(T)}
            row["delta_vs_baseline"] = {names[t]: float(scores[i, t] - base[i, t]) for t in range(T)}
        if params.get("return_tracks", True):
            row["tracks"] = {names[t]: np.round(_downsample(preds[i, t], track_bins), 5).tolist() for t in range(T)}
        rows.append(row)

    out_dir = Path(params.get("output_dir", "runs/predict"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "length", "gc_content"] + [f"score_{t}" for t in names])
        for r in rows:
            wr.writerow([r["id"], r["length"], round(r["gc_content"], 4)] + [r["scores"][t] for t in names])

    cards = []
    bad = qc["n_padded"] + qc["n_cropped"] + qc["n_invalid_chars"]
    cards.append(card("Input QC", "warn" if bad else "pass",
                      f"{qc['n_sequences']} sequences, lengths {qc['length_min']}-{qc['length_max']} bp "
                      f"(model expects {L}), mean GC {qc['mean_gc']:.2f}."))
    corrs = [c for c in (pearson(scores[:, t], rc_scores[:, t]) for t in range(T)) if c is not None]
    if corrs:
        med = float(np.median(corrs))
        cards.append(card("Reverse-complement consistency", "pass" if med >= 0.8 else "warn",
                          f"Median correlation between forward and reverse-complement scores across sequences: {med:.2f}."))
    else:
        cards.append(card("Reverse-complement consistency", "info", "Needs at least 3 sequences with varying scores."))
    if base is not None:
        frac = float((scores > base).mean())
        cards.append(card("Shuffled baseline", "info",
                          f"{frac:.0%} of sequence/task scores exceed their dinucleotide-shuffled baseline "
                          f"(mean of {n_sh} shuffles). Shuffles keep composition but destroy motifs."))
    cards.append(card("Scope", "info", "These are checks, not proof. Passing them does not show the biology is right."))

    return {"model": {**adapter.meta, "seq_len": L, "tasks": names}, "device": device,
            "results": {"predictions": rows, "qc": qc, "csv": str(csv_path)},
            "evidence": cards, "warnings": warnings}