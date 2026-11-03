import random
from pathlib import Path

import numpy as np

from core.device import detect_device
from core.models import load_model
from core.sequences import BASES, aggregate, dinuc_shuffle, fit_with_offset
from core.utils import card


def handle_design_workflow(params: dict) -> dict:
    """
    params:
      model, device, seeds (list of strings) OR n_random, on_targets [idx], off_targets [idx],
      center_bins, max_iter(20), candidates_per_iter(64), patience(3), n_baseline(20),
      seed, output_dir
    """
    device = detect_device(params.get("device", "auto"))
    adapter, warnings = load_model(params["model"], device, params.get("batch_size"))
    L, T = adapter.seq_len, len(adapter.task_names)
    names = adapter.task_names
    on = [int(i) for i in params.get("on_targets", [0])]
    off = [int(i) for i in params.get("off_targets", [])]
    if not on or any(i < 0 or i >= T for i in on + off):
        raise ValueError(f"target indices must be within 0..{T - 1}")
    center = params.get("center_bins", 0)
    K = int(params.get("candidates_per_iter", 64))
    max_iter = int(params.get("max_iter", 20))
    patience = int(params.get("patience", 3))
    n_base = int(params.get("n_baseline", 20))
    seed = int(params.get("seed", 0))
    rng, rnd = np.random.default_rng(seed), random.Random(seed)

    seeds = params.get("seeds")
    if not seeds:
        seeds = ["".join(rnd.choice(BASES) for _ in range(L)) for _ in range(int(params.get("n_random", 2)))]
    seeds = [fit_with_offset(s.upper(), L)[0] for s in seeds]

    def objective(seqs):
        agg = aggregate(adapter.predict(seqs), center)
        score = agg[:, on].mean(axis=1) - (agg[:, off].mean(axis=1) if off else 0.0)
        return np.asarray(score), agg

    designs = []
    for si, seq0 in enumerate(seeds):
        cur = seq0
        sc, agg = objective([cur])
        cur_score, start_agg = float(sc[0]), agg[0]
        history = [{"iter": 0, "score": cur_score, "mutation": None}]
        mutations, stale = [], 0
        for it in range(1, max_iter + 1):
            cands, moves = [], []
            for p in rng.integers(0, L, K):
                p = int(p)
                b = [x for x in BASES if x != cur[p]][int(rng.integers(3))]
                cands.append(cur[:p] + b + cur[p + 1:])
                moves.append((p, cur[p], b))
            scores, _ = objective(cands)
            j = int(np.argmax(scores))
            if scores[j] > cur_score + 1e-9:
                cur, cur_score = cands[j], float(scores[j])
                mutations.append({"pos_1based": moves[j][0] + 1, "from": moves[j][1], "to": moves[j][2]})
                history.append({"iter": it, "score": cur_score, "mutation": mutations[-1]})
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        _, end_agg = objective([cur])
        end_agg = end_agg[0]

        base = np.array([])
        if n_base > 0:
            base, _ = objective([dinuc_shuffle(seq0, rnd) for _ in range(n_base)])
        pct = float((base < cur_score).mean() * 100) if len(base) else None
        designs.append({
            "seed_index": si, "seed_sequence": seq0, "designed_sequence": cur,
            "n_mutations": len(mutations), "mutations": mutations,
            "score_before": float(sc[0]), "score_after": cur_score,
            "task_scores_before": {names[t]: float(start_agg[t]) for t in range(T)},
            "task_scores_after": {names[t]: float(end_agg[t]) for t in range(T)},
            "history": history,
            "baseline_mean": float(base.mean()) if len(base) else None,
            "baseline_std": float(base.std()) if len(base) else None,
            "percentile_vs_shuffled": pct,
        })

    out_dir = Path(params.get("output_dir", "runs/design"))
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta = out_dir / "designed.fasta"
    with open(fasta, "w") as f:
        for d in designs:
            f.write(f">design_{d['seed_index']}_score_{d['score_after']:.4f}\n{d['designed_sequence']}\n")

    improved = sum(d["score_after"] > d["score_before"] for d in designs)
    cards = [card("Objective improvement", "pass" if improved == len(designs) else "warn",
                  f"{improved}/{len(designs)} designs improved over their seed.")]
    pcts = [d["percentile_vs_shuffled"] for d in designs if d["percentile_vs_shuffled"] is not None]
    if pcts:
        cards.append(card("Shuffled-seed baseline", "pass" if min(pcts) >= 95 else "warn",
                          f"Designs scored above {min(pcts):.0f}% (worst case) of dinucleotide-shuffled seeds."))
    cards.append(card("Same-model caveat", "info",
                      "Designs are optimised against the same model that scores them. Validate with a second model."))

    return {"model": {**adapter.meta, "seq_len": L, "tasks": names}, "device": device,
            "results": {"designs": designs, "fasta": str(fasta)}, "evidence": cards, "warnings": warnings}