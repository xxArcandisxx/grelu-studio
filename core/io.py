import numpy as np
import csv

from core.sequences import BASES
from core.dna_inputs import normalize_dna, parse_records


def _clean(s) -> str:
    return normalize_dna(s)


def read_fasta(path):
    with open(path, encoding='utf-8-sig') as stream:
        records = parse_records(stream.read(), max_records=1000, allow_raw=False)
    return [(row['id'],row['sequence']) for row in records]


def read_sequences(params: dict):
    """Accepts ONE of: 'sequences' (list of strings or {id, sequence}), 'fasta' (path),
    'sequences_csv' (path with a 'sequence' column and optional 'id').
    Returns ([(id, sequence), ...], warnings)."""
    sources = [key for key in ['sequences','fasta','sequences_csv'] if key in params and params[key] is not None]
    if len(sources) != 1:
        raise ValueError('Provide exactly one sequence source; inputs are not silently selected or dropped.')
    seqs = []
    if sources[0] == 'sequences':
        if not isinstance(params['sequences'], (list,tuple)) or not params['sequences']:
            raise ValueError('sequences must be a nonempty list of DNA strings or records.')
        for i, s in enumerate(params["sequences"]):
            if isinstance(s, dict):
                if 'sequence' not in s:
                    raise ValueError('Each sequence record needs a sequence field.')
                seqs.append((s.get("id", f"seq{i}"), _clean(s["sequence"])))
            else:
                seqs.append((f"seq{i}", _clean(s)))
    elif sources[0] == 'fasta':
        seqs = read_fasta(params["fasta"])
    elif sources[0] == 'sequences_csv':
        with open(params['sequences_csv'], encoding='utf-8-sig', newline='') as stream:
            reader=csv.DictReader(stream, strict=True)
            headers=[key.lower().strip() for key in reader.fieldnames or []]
            if 'sequence' not in headers or len(headers)!=len(set(headers)):
                raise ValueError("sequences_csv needs unique headers including 'sequence'.")
            reader.fieldnames=headers
            for i,row in enumerate(reader):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError('Sequence CSV contains missing or extra fields.')
                seqs.append((row.get('id',f'seq{i}'),_clean(row['sequence'])))
    else:
        raise ValueError("Provide one of: 'sequences', 'fasta', 'sequences_csv'.")

    warnings = []
    if not seqs:
        raise ValueError("No sequences found in the input.")
    cap = int(params.get("max_sequences", 1000))
    if len(seqs) > cap:
        raise ValueError(f'Input has {len(seqs)} records; max_sequences={cap}. Records are not truncated.')
    ids=[identifier for identifier,_ in seqs]
    if any(not isinstance(identifier,str) or not identifier for identifier in ids) or len(ids)!=len(set(ids)):
        raise ValueError('Sequence identifiers must be nonempty, distinct strings.')
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
