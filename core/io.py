import numpy as np

from core.sequences import BASES


def _clean(s) -> str:
    return "".join(str(s).split()).upper()


def read_fasta(path):
    out, name, buf = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    out.append((name, _clean("".join(buf))))
                parts = line[1:].split()
                name = parts[0] if parts else f"seq{len(out)}"
                buf = []
            else:
                buf.append(line)
    if name is not None:
        out.append((name, _clean("".join(buf))))
    return out


def read_sequences(params: dict):
    """Accepts ONE of: 'sequences' (list of strings or {id, sequence}), 'fasta' (path),
    'sequences_csv' (path with a 'sequence' column and optional 'id').
    Returns ([(id, sequence), ...], warnings)."""
    seqs = []
    if params.get("sequences"):
        for i, s in enumerate(params["sequences"]):
            if isinstance(s, dict):
                seqs.append((str(s.get("id", f"seq{i}")), _clean(s["sequence"])))
            else:
                seqs.append((f"seq{i}", _clean(s)))
    elif params.get("fasta"):
        seqs = read_fasta(params["fasta"])
    elif params.get("sequences_csv"):
        import pandas as pd

        df = pd.read_csv(params["sequences_csv"])
        df.columns = [c.lower().strip() for c in df.columns]
        if "sequence" not in df.columns:
            raise ValueError("sequences_csv needs a 'sequence' column.")
        ids = df["id"].astype(str).tolist() if "id" in df.columns else [f"seq{i}" for i in range(len(df))]
        seqs = [(i, _clean(s)) for i, s in zip(ids, df["sequence"])]
    else:
        raise ValueError("Provide one of: 'sequences', 'fasta', 'sequences_csv'.")

    warnings = []
    before = len(seqs)
    seqs = [(i, s) for i, s in seqs if s]
    if len(seqs) < before:
        warnings.append(f"{before - len(seqs)} empty sequence(s) dropped.")
    if not seqs:
        raise ValueError("No sequences found in the input.")
    cap = int(params.get("max_sequences", 1000))
    if len(seqs) > cap:
        warnings.append(f"Input had {len(seqs)} sequences; only the first {cap} are used (max_sequences).")
        seqs = seqs[:cap]
    return seqs, warnings


def sequence_qc(seqs, L: int):
    """Input QC shown to the user before/after a run. Returns (qc_dict, warnings)."""
    lens = np.array([len(s) for _, s in seqs])
    gcs, invalid, with_n = [], 0, 0
    for _, s in seqs:
        acgt = sum(s.count(b) for b in BASES)
        n_cnt = s.count("N")
        invalid += len(s) - acgt - n_cnt
        with_n += int(n_cnt > 0)
        gcs.append((s.count("G") + s.count("C")) / max(acgt, 1))
    qc = {
        "n_sequences": len(seqs),
        "length_min": int(lens.min()),
        "length_median": float(np.median(lens)),
        "length_max": int(lens.max()),
        "model_input_length": int(L),
        "n_cropped": int((lens > L).sum()),
        "n_padded": int((lens < L).sum()),
        "n_with_N": int(with_n),
        "n_invalid_chars": int(invalid),
        "mean_gc": float(np.mean(gcs)),
    }
    warnings = []
    if qc["n_padded"]:
        warnings.append(f"{qc['n_padded']} sequence(s) shorter than the model input ({L} bp) were padded with N.")
    if qc["n_cropped"]:
        warnings.append(f"{qc['n_cropped']} sequence(s) longer than {L} bp were center-cropped.")
    if qc["n_invalid_chars"]:
        warnings.append(f"{qc['n_invalid_chars']} character(s) outside A/C/G/T/N were treated as unknown (all zeros).")
    return qc, warnings