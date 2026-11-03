import csv
import random
from pathlib import Path

import numpy as np

from core.device import detect_device
from core.models import load_model
from core.sequences import BASES, aggregate, dinuc_shuffle, fit_with_offset, revcomp
from core.utils import card, log


def _fetch_context(fa, chrom, pos1, L):
    chrom = str(chrom)
    if chrom not in fa:
        alt = chrom[3:] if chrom.startswith("chr") else "chr" + chrom
        if alt not in fa:
            raise ValueError(f"chromosome {chrom} not found in genome fasta")
        chrom = alt
    clen = len(fa[chrom])
    start = pos1 - 1 - L // 2
    end = start + L
    s0, e0 = max(start, 0), min(end, clen)
    seq = str(fa[chrom][s0:e0]).upper()
    seq = "N" * (s0 - start) + seq + "N" * (end - e0)
    return seq, pos1 - 1 - start


def handle_vep_workflow(params: dict) -> dict:
    """
    params:
      model, device("auto"), variants_csv, genome_fasta (only if CSV has chrom/pos),
      output_dir, center_bins(0=all), n_shuffles(10), seed, batch_size, max_variants
    variants_csv columns (case-insensitive), either:
      A) variant_id, sequence, pos, ref, alt     (pos = 1-based inside the sequence)
      B) variant_id, chrom, pos, ref, alt        (pos = 1-based genomic; needs genome_fasta)
    """
    import pandas as pd

    device = detect_device(params.get("device", "auto"))
    adapter, warnings = load_model(params["model"], device, params.get("batch_size"))
    L, T = adapter.seq_len, len(adapter.task_names)
    center = params.get("center_bins", 0)
    n_sh = int(params.get("n_shuffles", 10))
    rnd = random.Random(int(params.get("seed", 0)))

    df = pd.read_csv(params["variants_csv"])
    df.columns = [c.lower().strip() for c in df.columns]
    if params.get("max_variants"):
        df = df.head(int(params["max_variants"]))
    has_seq = "sequence" in df.columns
    fa = None
    if not has_seq:
        if not params.get("genome_fasta"):
            raise ValueError("CSV has no 'sequence' column, so 'genome_fasta' is required.")
        from pyfaidx import Fasta

        fa = Fasta(params["genome_fasta"])

    records, ref_seqs, alt_seqs, positions, keep_idx = [], [], [], [], []
    for i, row in df.iterrows():
        vid = str(row["variant_id"]) if "variant_id" in df.columns else f"var{i}"
        ref, alt = str(row["ref"]).upper(), str(row["alt"]).upper()
        rec = {"variant_id": vid, "ref": ref, "alt": alt}
        records.append(rec)
        if len(ref) != 1 or len(alt) != 1 or ref not in BASES or alt not in BASES:
            rec["status"] = "skipped_not_snv"
            continue
        try:
            if has_seq:
                seq, shift = fit_with_offset(str(row["sequence"]).upper(), L)
                p = int(row["pos"]) - 1 + shift
            else:
                seq, p = _fetch_context(fa, row["chrom"], int(row["pos"]), L)
        except Exception as e:  # noqa: BLE001
            rec["status"] = f"error: {e}"
            continue
        if not 0 <= p < L:
            rec["status"] = "skipped_outside_model_window"
            continue
        if seq[p] != ref:
            rec["status"] = f"skipped_ref_mismatch (sequence has {seq[p]})"
            continue
        rec["status"] = "ok"
        ref_seqs.append(seq)
        alt_seqs.append(seq[:p] + alt + seq[p + 1:])
        positions.append(p)
        keep_idx.append(len(records) - 1)

    n = len(ref_seqs)
    if n == 0:
        raise ValueError("No usable variants after validation. Check the 'status' column / allele match.")
    skipped = len(records) - n
    if skipped:
        warnings.append(f"{skipped} variant(s) skipped; see 'status' for each.")

    log(f"[engine] VEP: {n} variants on {device}")
    pr = adapter.predict(ref_seqs + alt_seqs)
    ref_s, alt_s = aggregate(pr[:n], center), aggregate(pr[n:], center)
    effect = alt_s - ref_s  # (n, T)

    # Evidence 1: reverse-complement consistency
    prc = adapter.predict([revcomp(s) for s in ref_seqs] + [revcomp(s) for s in alt_seqs])
    eff_rc = aggregate(prc[n:], center) - aggregate(prc[:n], center)
    rc_agree = (np.sign(effect) == np.sign(eff_rc)).mean(axis=1)

    # Evidence 2: dinucleotide-shuffled background p-values
    pvals = np.full((n, T), np.nan)
    if n_sh > 0:
        for k in range(n):
            p, ref, alt = positions[k], ref_seqs[k][positions[k]], alt_seqs[k][positions[k]]
            bg_r, bg_a = [], []
            for _ in range(n_sh):
                sh = dinuc_shuffle(ref_seqs[k], rnd)
                bg_r.append(sh[:p] + ref + sh[p + 1:])
                bg_a.append(sh[:p] + alt + sh[p + 1:])
            pb = adapter.predict(bg_r + bg_a)
            eff_bg = aggregate(pb[n_sh:], center) - aggregate(pb[:n_sh], center)
            pvals[k] = (1 + (np.abs(eff_bg) >= np.abs(effect[k])[None, :]).sum(axis=0)) / (1 + n_sh)

    names = adapter.task_names
    for j, k in enumerate(keep_idx):
        top = int(np.argmax(np.abs(effect[j])))
        records[k].update({
            "top_task": names[top],
            "top_effect": float(effect[j, top]),
            "effects": {names[t]: float(effect[j, t]) for t in range(T)},
            "ref_scores": {names[t]: float(ref_s[j, t]) for t in range(T)},
            "alt_scores": {names[t]: float(alt_s[j, t]) for t in range(T)},
            "rc_sign_agreement": float(rc_agree[j]),
            "pvalues": None if n_sh <= 0 else {names[t]: float(pvals[j, t]) for t in range(T)},
            "top_task_pvalue": None if n_sh <= 0 else float(pvals[j, top]),
        })

    out_dir = Path(params.get("output_dir", "runs/vep"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "vep_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant_id", "ref", "alt", "status", "top_task", "top_effect",
                    "top_task_pvalue", "rc_sign_agreement"] + [f"effect_{t}" for t in names])
        for r in records:
            eff = r.get("effects", {})
            w.writerow([r["variant_id"], r["ref"], r["alt"], r["status"], r.get("top_task", ""),
                        r.get("top_effect", ""), r.get("top_task_pvalue", ""),
                        r.get("rc_sign_agreement", "")] + [eff.get(t, "") for t in names])

    med = float(np.median(rc_agree))
    cards = [card("Reverse-complement consistency", "pass" if med >= 0.8 else "warn",
                  f"Median sign agreement between forward and reverse-complement effects: {med:.2f}.")]
    if n_sh > 0:
        sig = float(np.mean([r["top_task_pvalue"] <= 0.05 for r in records if r["status"] == "ok"]))
        cards.append(card("Shuffled-background comparison", "info",
                          f"{sig:.0%} of variants beat the shuffled background (p<=0.05) on their top task. "
                          f"With {n_sh} shuffles the smallest possible p-value is {1/(1+n_sh):.3f}; use 100+ for real p-values."))
    cards.append(card("Scope", "info", "These are checks, not proof. Passing them does not show the biology is right."))

    return {"model": {**adapter.meta, "seq_len": L, "tasks": names}, "device": device,
            "results": {"variants": records, "csv": str(csv_path)}, "evidence": cards, "warnings": warnings}